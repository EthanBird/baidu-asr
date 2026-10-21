# -*- coding: utf-8 -*-
"""
零依赖百度输入法 ASR 二进制帧与协议编解码模块 (`baidu_asr.codec`)：
1. 百度 `vse.baidu.com/ws/asr` 帧协议：
   - 外层帧：`4B 小端长度 (<I)` + `4B 协议头 (03 01 00 <type>)` + `载荷`
   - 客户端帧类型：
     - `03 01 00 00`：初始化参数帧（`CTRL` 10字节头 + `zlib` raw deflate 压缩 JSON）
     - `03 01 00 01`：音频数据帧（首包附带 `44 00 00 00`，每包携带 2 个 88B 定长 Opus Record）
     - `03 01 00 03`：音频结束帧 (EOF)
   - 服务端帧类型：
     - `03 01 00 f4`：服务端参数就绪确认帧 (Ready ACK)
     - `03 01 00 f8`：服务端 VAD 状态帧
     - `03 01 00 f0`：流式增量识别结果帧 (`partial`)
     - `03 01 00 fa`：句末初判帧
     - `03 01 00 f1`：最终候选识别结果帧 (`final`)
     - `03 01 00 f3`：会话结束确认帧 (Session End ACK)
2. RFC 6455 WebSocket 零拷贝全零掩码帧打包 (`pack_ws_frame`)
3. 流式增量文本差分计算器 (`compute_delta`)
"""

from __future__ import annotations

import json
import struct
import zlib
from typing import Any, Dict, List, Optional, Tuple, Union

RATE = 16000
CTRL = bytes.fromhex("757b0800000000000003")
RECORD_PREFIX = bytes.fromhex("0000005001000000")
FIRST_AUDIO_HEAD = bytes([3, 1, 0, 1]) + bytes.fromhex("44000000")
NEXT_AUDIO_HEAD = bytes([3, 1, 0, 1])
EOF_BODY = bytes([3, 1, 0, 3])

# 服务端帧类型常量
MSG_TYPE_PARTIAL = 0xF0
MSG_TYPE_FINAL = 0xF1
MSG_TYPE_SESSION_END = 0xF3
MSG_TYPE_READY_ACK = 0xF4
MSG_TYPE_VAD_STATUS = 0xF8
MSG_TYPE_PRE_FINAL = 0xFA

TEMPLATE: Dict[str, Any] = {
    "key": "com.baidu.input_nlu",
    "wwd": "",
    "ctl": '{"s_wake":true,"s_sdk_vad_mode":1,"s_link_switch":"ws"}',
    "cuid": "FUUIDC9F8E34671E37FC0E902CC886AD|VOFCAOU7Z",
    "pfm": "Android&sdk_gphone_x86_64&11&30&3&FUUIDC9F8E34671E37FC0E902CC886AD|VOFCAOU7Z&3",
    "ver": "4.8.3.285:Core_3.0.0",
    "pid": 597,
    "smr": RATE,
    "enable_combined_tts": 1,
    "sn": "",
    "app": "com.baidu.input",
    "cid": 0,
    "prop_list": [10005],
    "dep": '{"pkg":"com.android.settings.intelligence"}',
    "masr": 1,
    "fun": 537198592,
}


def frame(body: bytes) -> bytes:
    """打包百度 4B 小端长度前缀帧"""
    return struct.pack("<I", len(body)) + body


def record(pkt: bytes) -> bytes:
    """打包单个 20ms CBR Opus Record (8B 头部 + 80B 负载 = 88B)"""
    return RECORD_PREFIX + pkt[:80].ljust(80, b"\x00")


def param_frame(
    sn: str,
    *,
    pid: int = 597,
    cuid: Optional[str] = None,
    prop_list: Optional[List[int]] = None,
    fun: int = 537198592,
    extra_params: Optional[Dict[str, Any]] = None,
) -> bytes:
    """构造百度 ASR 会话初始化参数帧 (`03 01 00 00` + CTRL + raw deflate JSON)"""
    js = dict(TEMPLATE, sn=sn, pid=int(pid), fun=int(fun))
    if cuid:
        js["cuid"] = cuid
    if prop_list is not None:
        js["prop_list"] = list(prop_list)
    if extra_params:
        js.update(extra_params)
    raw = json.dumps(js, ensure_ascii=False).encode("utf-8")
    co = zlib.compressobj(9, 8, -15)
    body = bytes([3, 1, 0, 0]) + CTRL + co.compress(raw) + co.flush()
    return frame(body)


def eof_frame() -> bytes:
    """构造百度 ASR 音频结束帧 (`03 01 00 03`)"""
    return frame(EOF_BODY)


def pack_ws_frame(payload: Union[bytes, str], opcode: int = 2) -> bytes:
    """
    打包 RFC 6455 客户端到服务端 WebSocket 帧（采用 4 字节全零 Masking-Key 免除逐字节 XOR 开销）
    """
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    n = len(raw)
    if n < 126:
        return struct.pack("!BB4s", 0x80 | opcode, 0x80 | n, b"\x00\x00\x00\x00") + raw
    if n < 65536:
        return struct.pack("!BBH4s", 0x80 | opcode, 0x80 | 126, n, b"\x00\x00\x00\x00") + raw
    return struct.pack("!BBQ4s", 0x80 | opcode, 0x80 | 127, n, b"\x00\x00\x00\x00") + raw


def parse_baidu_message(msg: bytes) -> Tuple[int, Optional[Dict[str, Any]]]:
    """
    解析百度服务端返回的二进制消息，返回 `(msg_type_byte, parsed_json_or_none)`。
    """
    if not msg or len(msg) < 8:
        return 0, None
    msg_type = msg[7]
    if len(msg) == 8:
        return msg_type, None
    txt = msg[8:].decode("utf-8", "ignore").strip()
    if not txt.startswith("{"):
        return msg_type, None
    try:
        return msg_type, json.loads(txt)
    except Exception:
        return msg_type, None


_PUNC_SET = frozenset("，。！？、；：,.!?;: ")


def compute_delta(prev: str, curr: str) -> str:
    """计算流式识别相邻两帧文本之间的新增尾部内容 (delta)"""
    if not curr:
        return ""
    if not prev:
        return curr
    if curr.startswith(prev):
        return curr[len(prev) :]
    i, j = 0, 0
    len_p, len_c = len(prev), len(curr)
    while i < len_p and j < len_c:
        if prev[i] == curr[j]:
            i += 1
            j += 1
        elif i + 1 < len_p and j + 1 < len_c and prev[i + 1] == curr[j + 1]:
            i += 2
            j += 2
        elif prev[i] in _PUNC_SET:
            i += 1
        elif curr[j] in _PUNC_SET:
            j += 1
        else:
            break
    return curr[j:] if j > 0 else curr

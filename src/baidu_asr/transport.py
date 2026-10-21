# -*- coding: utf-8 -*-
"""
零依赖百度输入法 ASR 原生标准库 TLS WebSocket 传输层 (`baidu_asr.transport`)：
1. `BaiduWSSConnection`：基于标准库 `ssl` + `socket` + `select` 实现 RFC 6455 客户端，彻底移除 `websockets` 依赖。
2. 消除原版三大延迟瓶颈：
   - 以精确等待服务端 `03 01 00 f4` (Ready ACK) 替代原版的盲等 `asyncio.sleep(0.2)`；
   - 以单次 `sendall()` 合并写出全部 20ms 双 Record 帧 + EOF 帧替代原版的每 2 帧 `asyncio.sleep(0.04)`；
   - 以检测服务端 `03 01 00 f3` (Session End ACK) 立即退出替代原版末尾的 3 秒 `TimeoutError` 等待。
3. 内置可选豆包官方免鉴权 LLM「智能整理」(`organize_text`)，自动为百度 ASR 原文补齐标点、去水词并分段。
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import select
import socket
import ssl
import struct
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from .audio import encode_pcm_to_baidu_ws_frames
from .codec import (
    MSG_TYPE_FINAL,
    MSG_TYPE_PARTIAL,
    MSG_TYPE_PRE_FINAL,
    MSG_TYPE_SESSION_END,
    eof_frame,
    pack_ws_frame,
    param_frame,
    parse_baidu_message,
)

BAIDU_WS_HOST = "vse.baidu.com"
BAIDU_WS_PATH = "/ws/asr"

DOUBAO_ORGANIZE_HOST = "ime.oceancloudapi.com"
DOUBAO_ORGANIZE_PATH = "/api/v2/ai/text_organization"
DOUBAO_ORGANIZE_SCENE = 6
DOUBAO_UA = (
    "com.bytedance.android.doubaoime/100502008 "
    "(Linux; U; Android 16; zh_CN; Pixel 7 Pro; Build/BP2A.250605.031.A2)"
)


class BaiduWSSConnection:
    """原生标准库 RFC 6455 TLS WebSocket 连接（专用于 vse.baidu.com/ws/asr）"""

    def __init__(self, sn: str, ssl_ctx: ssl.SSLContext, timeout: float = 20.0):
        self.sn = sn
        self.timeout = timeout
        self.closed = False
        self._buf = bytearray()

        raw_sock = socket.create_connection((BAIDU_WS_HOST, 443), timeout=timeout)
        raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock: ssl.SSLSocket = ssl_ctx.wrap_socket(raw_sock, server_hostname=BAIDU_WS_HOST)
        self._handshake()

    def _handshake(self) -> None:
        ws_key = base64.b64encode(os.urandom(16)).decode("ascii")
        req = (
            f"GET {BAIDU_WS_PATH}?sn={self.sn} HTTP/1.1\r\n"
            f"Host: {BAIDU_WS_HOST}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {ws_key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        ).encode("ascii")
        self.sock.sendall(req)

        header_buf = bytearray()
        while b"\r\n\r\n" not in header_buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                self.close()
                raise ConnectionError("Baidu WebSocket 握手失败: 连接提前关闭")
            header_buf.extend(chunk)

        sep = header_buf.index(b"\r\n\r\n")
        head_bytes = bytes(header_buf[:sep])
        self._buf.extend(header_buf[sep + 4 :])
        status_line = head_bytes.splitlines()[0].decode("latin1", "ignore")
        if "101" not in status_line:
            self.close()
            raise ConnectionError(f"Baidu WebSocket 握手被拒绝: {status_line}")

    def _ensure_buffered(self, n: int) -> None:
        while len(self._buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                self.closed = True
                raise EOFError("WebSocket EOF")
            self._buf.extend(chunk)

    def send_binary(self, payload: bytes) -> None:
        self.sock.sendall(pack_ws_frame(payload, opcode=2))

    def send_raw(self, raw_frames: bytes) -> None:
        if raw_frames:
            self.sock.sendall(raw_frames)

    def wait_readable(self, wait_sec: float = 0.0) -> bool:
        if self._buf or self.sock.pending() > 0:
            return True
        r, _, _ = select.select([self.sock], [], [], max(0.0, wait_sec))
        return bool(r)

    def recv_message(self, timeout: Optional[float] = None) -> Optional[bytes]:
        if timeout is not None:
            self.sock.settimeout(timeout)
        try:
            while True:
                try:
                    self._ensure_buffered(2)
                except EOFError:
                    return None
                b0, b1 = self._buf[0], self._buf[1]
                opcode = b0 & 0x0F
                masked = (b1 & 0x80) != 0
                length = b1 & 0x7F
                hdr_len = 2
                if length == 126:
                    hdr_len += 2
                    self._ensure_buffered(hdr_len)
                    length = struct.unpack("!H", self._buf[2:4])[0]
                elif length == 127:
                    hdr_len += 8
                    self._ensure_buffered(hdr_len)
                    length = struct.unpack("!Q", self._buf[2:10])[0]

                mask_offset = hdr_len
                if masked:
                    hdr_len += 4
                total_len = hdr_len + length
                self._ensure_buffered(total_len)

                payload = bytes(self._buf[hdr_len:total_len])
                if masked and payload:
                    mask_key = self._buf[mask_offset : mask_offset + 4]
                    payload = bytes(b ^ mask_key[i & 3] for i, b in enumerate(payload))
                del self._buf[:total_len]

                if opcode == 9:
                    self.sock.sendall(pack_ws_frame(payload, opcode=10))
                    continue
                if opcode == 10:
                    continue
                if opcode == 8:
                    self.closed = True
                    return None
                return payload
        finally:
            if timeout is not None and not self.closed:
                try:
                    self.sock.settimeout(self.timeout)
                except Exception:
                    pass

    def drain_messages(self, initial_wait_sec: float = 0.0) -> List[bytes]:
        out: List[bytes] = []
        wait = initial_wait_sec
        while not self.closed and self.wait_readable(wait):
            wait = 0.0
            try:
                msg = self.recv_message(timeout=1.5)
                if msg is None:
                    break
                out.append(msg)
            except (socket.timeout, BlockingIOError):
                break
            except Exception:
                self.closed = True
                break
        return out

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                self.sock.sendall(pack_ws_frame(b"\x03\xe8", opcode=8))
            except Exception:
                pass
            try:
                self.sock.close()
            except Exception:
                pass


class BaiduTransport:
    """管理百度输入法 ASR 并发 WebSocket 转写与可选豆包 LLM 智能整理"""

    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout
        self._ssl_ctx = ssl.create_default_context()
        self._org_pool: List[http.client.HTTPSConnection] = []
        self._org_lock = threading.Lock()

    def open_session(
        self,
        *,
        pid: int = 597,
        cuid: Optional[str] = None,
        prop_list: Optional[List[int]] = None,
        fun: int = 537198592,
        extra_params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Tuple[BaiduWSSConnection, str]:
        """建立一条已完成参数握手的百度 ASR WebSocket 会话，返回 `(conn, sn)`"""
        sn = str(uuid.uuid4())
        conn = BaiduWSSConnection(sn, self._ssl_ctx, timeout=timeout or self.timeout)
        pframe = param_frame(
            sn,
            pid=pid,
            cuid=cuid,
            prop_list=prop_list,
            fun=fun,
            extra_params=extra_params,
        )
        conn.send_binary(pframe)
        # 等待服务端 8 字节 03 01 00 f4 Ready ACK（约 80~120ms，免除盲目 sleep(0.2)）
        ack = conn.recv_message(timeout=5.0)
        if ack is None:
            conn.close()
            raise ConnectionError("百度 ASR 参数握手后连接提前断开")
        msg_type, data = parse_baidu_message(ack)
        if msg_type == MSG_TYPE_FINAL and data and data.get("err_no", 0) != 0:
            conn.close()
            raise RuntimeError(f"百度 ASR 鉴权/参数错误 ({data.get('err_no')}): {data.get('err_msg')}")
        return conn, sn

    def transcribe_pcm_segment(
        self,
        pcm_bytes: bytes,
        *,
        pid: int = 597,
        cuid: Optional[str] = None,
        prop_list: Optional[List[int]] = None,
        fun: int = 537198592,
        extra_params: Optional[Dict[str, Any]] = None,
        on_partial: Optional[Callable[[str], None]] = None,
        retries: int = 2,
        timeout: float = 25.0,
    ) -> Dict[str, Any]:
        """
        识别单个 16kHz PCM 片段，返回包含 `text`, `alternatives`, `confidence`, `speak_speed`, `voice_power`, `sn` 的完整字典。
        """
        ws_audio_eof = encode_pcm_to_baidu_ws_frames(pcm_bytes, include_eof=True)
        if not ws_audio_eof:
            return {"text": "", "alternatives": [], "confidence": None, "speak_speed": 0.0, "voice_power": 0.0, "sn": ""}

        last_err: Optional[Exception] = None
        for attempt in range(max(1, int(retries))):
            conn: Optional[BaiduWSSConnection] = None
            try:
                conn, sn = self.open_session(
                    pid=pid,
                    cuid=cuid,
                    prop_list=prop_list,
                    fun=fun,
                    extra_params=extra_params,
                    timeout=timeout,
                )
                # 单次 sendall 发送所有 20ms 双 Record 音频帧 + EOF 帧
                conn.send_raw(ws_audio_eof)

                final_text = ""
                alternatives: List[str] = []
                confidence: Optional[int] = None
                speak_speed = 0.0
                voice_power = 0.0

                deadline = time.time() + timeout
                while time.time() < deadline:
                    raw_msg = conn.recv_message(timeout=max(0.2, deadline - time.time()))
                    if raw_msg is None:
                        break
                    msg_type, d = parse_baidu_message(raw_msg)
                    if msg_type == MSG_TYPE_SESSION_END:
                        break
                    if not d:
                        continue
                    err_no = int(d.get("err_no", 0))
                    if err_no != 0:
                        # -3005 表示该片段为纯背景音乐/静音（asr server not find effective speech），正常视为空文本
                        if err_no == -3005 or "not find effective speech" in str(d.get("err_msg", "")):
                            continue
                        raise RuntimeError(f"百度 ASR 错误 ({err_no}): {d.get('err_msg')}")

                    if isinstance(d.get("result"), dict):
                        words = [
                            (w if isinstance(w, str) else w.decode("utf-8", "ignore")).strip()
                            for w in (d["result"].get("word") or [])
                            if w
                        ]
                        confs = d["result"].get("confident") or []
                        if words:
                            final_text = words[0]
                            alternatives = words
                            if confs:
                                confidence = int(confs[0])
                            if on_partial:
                                on_partial(final_text)
                    elif d.get("llm_asr_result"):
                        final_text = str(d["llm_asr_result"]).strip()
                        if on_partial:
                            on_partial(final_text)
                    elif d.get("asr_result") and not final_text:
                        final_text = str(d["asr_result"]).strip()

                    if d.get("speak_speed") is not None:
                        speak_speed = float(d["speak_speed"])
                    if d.get("voice_power") is not None:
                        voice_power = float(d["voice_power"])

                conn.close()
                return {
                    "text": final_text,
                    "alternatives": alternatives,
                    "confidence": confidence,
                    "speak_speed": round(speak_speed, 2),
                    "voice_power": round(voice_power, 2),
                    "sn": sn,
                }
            except Exception as e:
                last_err = e
                if conn is not None:
                    conn.close()
                if attempt + 1 < retries:
                    time.sleep(0.25 * (attempt + 1))

        raise RuntimeError(f"百度 ASR 识别失败: {last_err}")

    # -------------------- 可选豆包官方 LLM「智能整理」 --------------------
    def _organize_one(self, text: str, timeout: float = 45.0) -> str:
        text = text.strip()
        if not text:
            return ""
        body = json.dumps(
            {
                "scene": DOUBAO_ORGANIZE_SCENE,
                "query": text,
                "space_at_cn_en_nb": 1,
                "space_at_newline": 0,
                "stream": False,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": DOUBAO_UA,
            "Connection": "keep-alive",
            "Content-Length": str(len(body)),
        }
        for attempt in range(2):
            conn: Optional[http.client.HTTPSConnection] = None
            with self._org_lock:
                if self._org_pool:
                    conn = self._org_pool.pop()
            if conn is None:
                conn = http.client.HTTPSConnection(DOUBAO_ORGANIZE_HOST, timeout=timeout, context=self._ssl_ctx)
                conn.connect()
            try:
                conn.request("POST", DOUBAO_ORGANIZE_PATH, body=body, headers=headers)
                resp = conn.getresponse()
                raw = resp.read()
                if resp.status == 200 and raw:
                    with self._org_lock:
                        if len(self._org_pool) < 12:
                            self._org_pool.append(conn)
                            conn = None
                    d = json.loads(raw.decode("utf-8", "ignore"))
                    if d.get("code") == 0:
                        return (d.get("data") or {}).get("content") or text
            except Exception:
                pass
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
        return text

    def organize_text(self, text: str, max_chunk_chars: int = 450, workers: int = 4) -> str:
        text = text.strip()
        if not text:
            return ""
        seps = "。！？!?\n；;，,"
        chunks: List[str] = []
        start, n = 0, len(text)
        while start < n:
            if start + max_chunk_chars >= n:
                tail = text[start:].strip()
                if tail:
                    chunks.append(tail)
                break
            end = start + max_chunk_chars
            split_pos = -1
            for sep in seps:
                pos = text.rfind(sep, start + max_chunk_chars // 2, end)
                if pos != -1:
                    split_pos = pos + 1
                    break
            if split_pos == -1:
                split_pos = end
            piece = text[start:split_pos].strip()
            if piece:
                chunks.append(piece)
            start = split_pos

        if len(chunks) == 1:
            return self._organize_one(chunks[0])
        actual_w = max(1, min(int(workers), len(chunks), 8))
        with ThreadPoolExecutor(max_workers=actual_w) as ex:
            parts = [p.strip() for p in ex.map(self._organize_one, chunks)]
        return "\n\n".join(p for p in parts if p)

    def close(self) -> None:
        with self._org_lock:
            conns = list(self._org_pool)
            self._org_pool.clear()
        for c in conns:
            try:
                c.close()
            except Exception:
                pass

# -*- coding: utf-8 -*-
"""
零依赖高性能百度 CBR Opus 编码与音频处理模块 (`baidu_asr.audio`)：
1. 直接通过 `ctypes` 绑定内置 `_bin/opus.dll`（或系统 `libopus`），彻底移除 `opuslib` 依赖。
2. 修复原版 `01-baidu_asr.py` 中 `step = RATE // 50 = 320` 字节切片导致 50% 帧重叠与 2 倍拉长失真的 Bug：
   16kHz 下 20ms (`320` samples) 对应精确的 **`640` 字节** 16-bit 单声道 PCM，每帧 CBR 32kbps 输出恰好 80 字节。
3. 支持在编码阶段将每 2 个 20ms Opus Record 直接预打包为百度 `[3, 1, 0, 1]` + RFC 6455 WebSocket 二进制帧流，
   彻底消除原版的 `asyncio.sleep(0.04)` 逐包延时，单次 `sendall()` 即可发完整个音频片段。
4. 使用 C 级 `audioop.rms` 自适应静音切分长音视频，并通过 `ffmpeg` 流式管道零临时文件解码。
"""

from __future__ import annotations

import array
import ctypes
import ctypes.util
import math
import os
import shutil
import subprocess
import wave
from typing import Iterable, Iterator, List, Optional, Tuple, Union

from .codec import FIRST_AUDIO_HEAD, NEXT_AUDIO_HEAD, eof_frame, frame, pack_ws_frame, record

try:
    import audioop as _audioop
except ImportError:
    _audioop = None  # Python 3.13+ 兜底

_OPUS_APPLICATION_RESTRICTED_LOWDELAY = 2051
_OPUS_SET_BITRATE_REQUEST = 4002
_OPUS_SET_VBR_REQUEST = 4006
_OPUS_SET_COMPLEXITY_REQUEST = 4010


def _load_libopus() -> Optional[ctypes.CDLL]:
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(pkg_dir, "_bin", "opus.dll"),
        os.path.join(pkg_dir, "opus.dll"),
    ]
    sys_lib = ctypes.util.find_library("opus") or ctypes.util.find_library("libopus-0")
    if sys_lib:
        candidates.append(sys_lib)
    candidates.extend(["opus.dll", "libopus.so.0", "libopus.0.dylib", "libopus.dylib"])

    for path in candidates:
        try:
            dll = ctypes.CDLL(path)
            dll.opus_encoder_create.argtypes = [
                ctypes.c_int32,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
            ]
            dll.opus_encoder_create.restype = ctypes.c_void_p

            dll.opus_encode.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int32,
            ]
            dll.opus_encode.restype = ctypes.c_int32

            dll.opus_encoder_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int32]
            dll.opus_encoder_ctl.restype = ctypes.c_int

            dll.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
            dll.opus_encoder_destroy.restype = None
            return dll
        except Exception:
            continue
    return None


_LIBOPUS = _load_libopus()


class BaiduOpusEncoder:
    """
    百度 `restricted_lowdelay` 32kbps CBR 20ms Opus 编码器
    （每 20ms = 320 samples = 640 bytes PCM -> 80 bytes Opus packet）
    """

    __slots__ = ("_st", "_out_buf", "_out_ptr", "_sent_first_pair")

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        bitrate: int = 32000,
        complexity: int = 10,
    ):
        if _LIBOPUS is None:
            raise RuntimeError("未找到 libopus / opus.dll")
        err = ctypes.c_int(0)
        st = _LIBOPUS.opus_encoder_create(
            sample_rate, channels, _OPUS_APPLICATION_RESTRICTED_LOWDELAY, ctypes.byref(err)
        )
        if err.value != 0 or not st:
            raise RuntimeError(f"opus_encoder_create 失败: code={err.value}")
        self._st: Optional[int] = st
        _LIBOPUS.opus_encoder_ctl(self._st, _OPUS_SET_BITRATE_REQUEST, int(bitrate))
        _LIBOPUS.opus_encoder_ctl(self._st, _OPUS_SET_VBR_REQUEST, 0)  # CBR = False VBR
        _LIBOPUS.opus_encoder_ctl(self._st, _OPUS_SET_COMPLEXITY_REQUEST, int(complexity))
        self._out_buf = ctypes.create_string_buffer(1024)
        self._out_ptr = ctypes.cast(self._out_buf, ctypes.c_char_p)
        self._sent_first_pair = False

    def encode_packets(self, pcm_bytes: bytes, pad_final: bool = True) -> List[bytes]:
        """将 16kHz PCM 编码为 20ms (80B) 裸 Opus 包列表（步长严格为 640 字节 = 320 采样点）"""
        if not pcm_bytes or not self._st:
            return []
        encode_fn = _LIBOPUS.opus_encode
        st = self._st
        out_buf = self._out_buf
        out_ptr = self._out_ptr
        total = len(pcm_bytes)
        pkts: List[bytes] = []
        off = 0

        while off + 640 <= total:
            n = encode_fn(st, pcm_bytes[off : off + 640], 320, out_ptr, 1024)
            if n < 0:
                raise RuntimeError(f"opus_encode 错误: {n}")
            pkts.append(out_buf.raw[:n])
            off += 640

        if pad_final and off < total:
            rem = pcm_bytes[off:] + b"\x00" * (640 - (total - off))
            n = encode_fn(st, rem, 320, out_ptr, 1024)
            if n < 0:
                raise RuntimeError(f"opus_encode 尾帧错误: {n}")
            pkts.append(out_buf.raw[:n])

        return pkts

    def encode_ws_chunk(self, pcm_bytes: bytes, include_eof: bool = False) -> bytes:
        """
        将一段 PCM 编码为百度双 Record 音频帧 (`[3, 1, 0, 1]`) 并打包为连续 RFC 6455 二进制帧流。
        自动跟踪首个双包是否需要附带 `44 00 00 00` 首包标记。
        """
        pkts = self.encode_packets(pcm_bytes, pad_final=True)
        ws_out = bytearray()
        n_pkts = len(pkts)
        i = 0
        while i < n_pkts:
            p1 = pkts[i]
            p2 = pkts[i + 1] if i + 1 < n_pkts else p1
            if not self._sent_first_pair:
                body = FIRST_AUDIO_HEAD + record(p1) + record(p2)
                self._sent_first_pair = True
            else:
                body = NEXT_AUDIO_HEAD + record(p1) + record(p2)
            ws_out.extend(pack_ws_frame(frame(body), opcode=2))
            i += 2

        if include_eof:
            ws_out.extend(pack_ws_frame(eof_frame(), opcode=2))
        return bytes(ws_out)

    def close(self) -> None:
        if self._st and _LIBOPUS is not None:
            _LIBOPUS.opus_encoder_destroy(self._st)
            self._st = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def opus_packets(pcm_bytes: bytes) -> List[bytes]:
    """将 16kHz PCM 编码为 20ms (80B) CBR Opus 包列表（兼容原版函数名，已修复 step=640B Bug）"""
    enc = BaiduOpusEncoder(16000, 1)
    try:
        return enc.encode_packets(pcm_bytes, pad_final=True)
    finally:
        enc.close()


def encode_pcm_to_baidu_ws_frames(pcm_bytes: bytes, include_eof: bool = True) -> bytes:
    """将 16kHz PCM 一次性编码并打包为百度双 Record WebSocket 二进制帧流"""
    enc = BaiduOpusEncoder(16000, 1)
    try:
        return enc.encode_ws_chunk(pcm_bytes, include_eof=include_eof)
    finally:
        enc.close()


# ============================================================================
# 2. C 级高速 VAD 能量计算与自适应静音切分
# ============================================================================
def compute_rms(pcm_bytes: bytes) -> float:
    n = (len(pcm_bytes) // 2) * 2
    if n == 0:
        return 0.0
    if _audioop is not None:
        return float(_audioop.rms(pcm_bytes[:n], 2))
    arr = array.array("h")
    arr.frombytes(pcm_bytes[:n])
    return math.sqrt(sum(x * x for x in arr) / max(1, len(arr)))


def split_pcm_by_silence(
    pcm_bytes: bytes,
    min_segment_sec: float = 10.0,
    max_segment_sec: float = 20.0,
) -> List[Tuple[float, float, bytes, float]]:
    win_bytes = 3200  # 100ms @ 16000Hz * 2 bytes
    total_len = len(pcm_bytes)
    if total_len == 0:
        return []

    n_wins = math.ceil(total_len / win_bytes)
    if _audioop is not None:
        rms_fn = _audioop.rms
        energies = [
            float(rms_fn(pcm_bytes[i * win_bytes : (i + 1) * win_bytes], 2))
            for i in range(n_wins)
        ]
    else:
        energies = [
            compute_rms(pcm_bytes[i * win_bytes : (i + 1) * win_bytes])
            for i in range(n_wins)
        ]

    min_w = max(10, int(round(min_segment_sec * 10)))
    max_w = max(min_w + 5, int(round(max_segment_sec * 10)))

    segments: List[Tuple[float, float, bytes, float]] = []
    cur = 0
    while cur < n_wins:
        if cur + max_w >= n_wins:
            end_w = n_wins
        else:
            end_w = min(range(cur + min_w, cur + max_w + 1), key=energies.__getitem__)
        b_start = cur * win_bytes
        b_end = min(total_len, end_w * win_bytes)
        sub_pcm = pcm_bytes[b_start:b_end]
        seg_energies = energies[cur:end_w]
        seg_rms = sum(seg_energies) / len(seg_energies) if seg_energies else 0.0
        segments.append((b_start / 32000.0, b_end / 32000.0, sub_pcm, seg_rms))
        cur = end_w

    return segments


# ============================================================================
# 3. 流式音频读取与 ffmpeg 管道解码
# ============================================================================
def iter_pcm16k_chunks(
    source: Union[str, os.PathLike, bytes, Iterable[bytes]],
    chunk_bytes: int = 32000,
    ffmpeg_path: Optional[str] = None,
) -> Iterator[bytes]:
    if isinstance(source, (bytes, bytearray)):
        if len(source) >= 12 and (
            source[:4] in (b"RIFF", b"OggS", b"fLaC", b"ID3\x03", b"ID3\x04")
            or source[4:8] == b"ftyp"
            or source[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")
        ):
            pcm_all = _decode_encoded_bytes_via_ffmpeg(bytes(source), ffmpeg_path=ffmpeg_path)
            for i in range(0, len(pcm_all), chunk_bytes):
                yield pcm_all[i : i + chunk_bytes]
            return
        for i in range(0, len(source), chunk_bytes):
            yield bytes(source[i : i + chunk_bytes])
        return

    if isinstance(source, os.PathLike):
        source = os.fspath(source)

    if not isinstance(source, str):
        for chunk in source:
            if chunk:
                yield bytes(chunk)
        return

    if not os.path.isfile(source):
        raise FileNotFoundError(f"找不到音频/视频文件: {source}")

    is_std_wav = False
    try:
        with wave.open(source, "rb") as w:
            if w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2:
                is_std_wav = True
                frames_per_chunk = max(320, chunk_bytes // 2)
                while True:
                    data = w.readframes(frames_per_chunk)
                    if not data:
                        break
                    yield data
        if is_std_wav:
            return
    except Exception:
        pass

    exe = ffmpeg_path or shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        exe,
        "-nostdin",
        "-v",
        "error",
        "-i",
        source,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    eof_reached = False
    try:
        while True:
            buf = proc.stdout.read(chunk_bytes)
            if not buf:
                eof_reached = True
                break
            yield buf
    finally:
        if not eof_reached:
            try:
                proc.terminate()
            except Exception:
                pass
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", errors="ignore").strip() if proc.stderr else ""
        if proc.stderr:
            proc.stderr.close()
        rc = proc.wait()
        if eof_reached and rc != 0 and err:
            raise RuntimeError(f"ffmpeg 流式解码失败 ({source}): {err}")


def _decode_encoded_bytes_via_ffmpeg(audio_bytes: bytes, ffmpeg_path: Optional[str] = None) -> bytes:
    if len(audio_bytes) >= 44 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        try:
            import io
            with wave.open(io.BytesIO(audio_bytes), "rb") as w:
                if w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2:
                    return w.readframes(w.getnframes())
        except Exception:
            pass
    exe = ffmpeg_path or shutil.which("ffmpeg") or "ffmpeg"
    proc = subprocess.run(
        [exe, "-v", "error", "-i", "pipe:0", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "pipe:1"],
        input=audio_bytes,
        capture_output=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 内存管道转码失败: {proc.stderr.decode('utf-8', 'ignore')[:300]}")
    return proc.stdout


def load_pcm16k(source: Union[str, os.PathLike, bytes], ffmpeg_path: Optional[str] = None) -> bytes:
    """一次性将任意音频/视频文件路径或字节流加载为完整 16kHz/16bit/mono PCM bytes"""
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
        if len(raw) >= 12 and (
            raw[:4] in (b"RIFF", b"OggS", b"fLaC", b"ID3\x03", b"ID3\x04")
            or raw[4:8] == b"ftyp"
            or raw[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")
        ):
            return _decode_encoded_bytes_via_ffmpeg(raw, ffmpeg_path=ffmpeg_path)
        return raw
    if isinstance(source, os.PathLike):
        source = os.fspath(source)
    if isinstance(source, str) and os.path.isfile(source):
        try:
            with wave.open(source, "rb") as w:
                if w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2:
                    return w.readframes(w.getnframes())
        except Exception:
            pass
    return b"".join(iter_pcm16k_chunks(source, chunk_bytes=131072, ffmpeg_path=ffmpeg_path))

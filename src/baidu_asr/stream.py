# -*- coding: utf-8 -*-
"""
百度输入法 ASR 协议级实时推流会话 (`BaiduStreamSession`)：
支持不断向 `feed(pcm_chunk)` 推入任意长度的 16kHz/16bit/单声道 PCM 字节流，
内部通过 `BaiduWSSConnection` 实时发送 20ms 双 Record Opus 帧并读取服务端 `03 01 00 f0` 增量响应，
实时产出 `partial` 与 `final` 事件，同时内置异常自愈回退与可选 LLM 智能整理。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

from .audio import BaiduOpusEncoder, compute_rms
from .codec import MSG_TYPE_SESSION_END, compute_delta, eof_frame, parse_baidu_message
from .models import Segment, StreamEvent, TranscriptResult
from .transport import BaiduWSSConnection

if TYPE_CHECKING:
    from .client import BaiduASR


class BaiduStreamSession:
    """协议级实时流式语音识别会话（Push Stream）"""

    def __init__(
        self,
        client: "BaiduASR",
        *,
        chunk_sec: float = 0.6,
        min_utterance_sec: float = 5.0,
        max_utterance_sec: float = 22.0,
        vad_silence_rms: float = 200.0,
        vad_silence_sec: float = 0.5,
        pid: int = 597,
        fun: int = 537198592,
        organize: bool = False,
        source_name: str = "<stream>",
    ):
        self.client = client
        self.chunk_sec = max(0.08, float(chunk_sec))
        self.min_utterance_sec = max(1.5, float(min_utterance_sec))
        self.max_utterance_sec = max(self.min_utterance_sec + 1.0, float(max_utterance_sec))
        self.vad_silence_rms = float(vad_silence_rms)
        self.vad_silence_sec = float(vad_silence_sec)
        self.pid = int(pid)
        self.fun = int(fun)
        self.organize = organize
        self.source_name = source_name

        # 每个双 Record 帧对应 40ms (1280 字节 PCM)
        raw_chunk = int(round(self.chunk_sec * 16000)) * 2
        self._chunk_bytes = max(1280, (raw_chunk // 1280) * 1280)

        self._pcm_buf = bytearray()
        self._total_pcm_bytes = 0
        self._t0 = time.perf_counter()

        self.segments: List[Segment] = []
        self._seg_index = 1
        self._utt_start_sec = 0.0
        self._utt_pcm_bytes = 0
        self._utt_pcm_data = bytearray()
        self._utt_silence_sec = 0.0
        self._utt_rms_sum = 0.0
        self._utt_rms_cnt = 0

        self._conn: Optional[BaiduWSSConnection] = None
        self._encoder: Optional[BaiduOpusEncoder] = None
        self._seq = 1
        self._current_text = ""
        self._last_partial_text = ""
        self._alternatives: List[str] = []
        self._confidence: Optional[int] = None
        self._speak_speed = 0.0
        self._voice_power = 0.0
        self._utt_done = False

    def __enter__(self) -> "BaiduStreamSession":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            for _ in self.flush():
                pass
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _reset_utterance(self, next_start_sec: float):
        if self._encoder is not None:
            self._encoder.close()
            self._encoder = None
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._seq = 1
        self._utt_start_sec = next_start_sec
        self._utt_pcm_bytes = 0
        self._utt_pcm_data.clear()
        self._utt_silence_sec = 0.0
        self._utt_rms_sum = 0.0
        self._utt_rms_cnt = 0
        self._current_text = ""
        self._last_partial_text = ""
        self._alternatives = []
        self._confidence = None
        self._speak_speed = 0.0
        self._voice_power = 0.0
        self._utt_done = False

    def _ensure_utterance_started(self):
        if self._conn is not None and not self._conn.closed:
            return
        conn, _ = self.client.transport.open_session(pid=self.pid, fun=self.fun)
        self._conn = conn
        self._encoder = BaiduOpusEncoder(16000, 1)

    @property
    def _history_text(self) -> str:
        return "".join(s.text for s in self.segments if s.text.strip())

    def _consume_raw(self, raw_msg: bytes) -> bool:
        msg_type, d = parse_baidu_message(raw_msg)
        if msg_type == MSG_TYPE_SESSION_END:
            self._utt_done = True
            return True
        if not d:
            return False
        if isinstance(d.get("result"), dict):
            words = [
                (w if isinstance(w, str) else w.decode("utf-8", "ignore")).strip()
                for w in (d["result"].get("word") or [])
                if w
            ]
            confs = d["result"].get("confident") or []
            if words:
                self._current_text = words[0]
                self._alternatives = words
                if confs:
                    self._confidence = int(confs[0])
        elif d.get("llm_asr_result"):
            self._current_text = str(d["llm_asr_result"]).strip()
        elif d.get("asr_result") and not self._current_text:
            self._current_text = str(d["asr_result"]).strip()

        if d.get("speak_speed") is not None:
            self._speak_speed = float(d["speak_speed"])
        if d.get("voice_power") is not None:
            self._voice_power = float(d["voice_power"])
        return False

    def _send_chunk(self, pcm_chunk: bytes) -> Optional[StreamEvent]:
        rms = compute_rms(pcm_chunk)
        chunk_dur = len(pcm_chunk) / 32000.0
        self._utt_pcm_bytes += len(pcm_chunk)
        self._utt_pcm_data.extend(pcm_chunk)
        self._utt_rms_sum += rms
        self._utt_rms_cnt += 1

        if rms < self.vad_silence_rms:
            self._utt_silence_sec += chunk_dur
        else:
            self._utt_silence_sec = 0.0

        self._ensure_utterance_started()
        assert self._encoder is not None and self._conn is not None

        ws_frames = self._encoder.encode_ws_chunk(pcm_chunk, include_eof=False)
        self._conn.send_raw(ws_frames)
        cur_seq = self._seq
        self._seq += 1

        for raw_msg in self._conn.drain_messages(initial_wait_sec=0.05):
            self._consume_raw(raw_msg)

        t = self._current_text
        if t and t != self._last_partial_text:
            delta = compute_delta(self._last_partial_text, t)
            self._last_partial_text = t
            cur_end = self._utt_start_sec + (self._utt_pcm_bytes / 32000.0)
            return StreamEvent(
                type="partial",
                segment_index=self._seg_index,
                seq=cur_seq,
                start=round(self._utt_start_sec, 2),
                end=round(cur_end, 2),
                text=t,
                delta=delta,
                full_text=self._history_text + t,
                is_final=False,
            )
        return None

    def _finalize_utterance(self) -> Optional[StreamEvent]:
        if self._utt_pcm_bytes == 0:
            return None

        if self._conn is not None and not self._conn.closed and not self._utt_done:
            try:
                self._conn.send_binary(eof_frame())
                deadline = time.time() + 10.0
                while time.time() < deadline:
                    raw_msg = self._conn.recv_message(timeout=max(0.2, deadline - time.time()))
                    if raw_msg is None:
                        break
                    if self._consume_raw(raw_msg):
                        break
            except Exception:
                pass

        best = self._current_text or self._last_partial_text
        avg_rms = self._utt_rms_sum / max(1, self._utt_rms_cnt)
        if not best and avg_rms >= 15.0 and self._utt_pcm_data:
            fallback = self.client.transport.transcribe_pcm_segment(
                bytes(self._utt_pcm_data),
                pid=self.pid,
                fun=self.fun,
            )
            best = fallback.get("text", "")
            self._alternatives = fallback.get("alternatives") or []
            self._confidence = fallback.get("confidence")
            self._speak_speed = float(fallback.get("speak_speed") or 0.0)
            self._voice_power = float(fallback.get("voice_power") or 0.0)

        cur_end = self._utt_start_sec + (self._utt_pcm_bytes / 32000.0)
        delta = compute_delta(self._last_partial_text, best)
        seg = Segment(
            index=self._seg_index,
            start=round(self._utt_start_sec, 2),
            end=round(cur_end, 2),
            text=best,
            rms=round(avg_rms, 1),
            confidence=self._confidence,
            speak_speed=round(self._speak_speed, 2),
            voice_power=round(self._voice_power, 2),
            alternatives=list(self._alternatives),
        )
        self.segments.append(seg)
        ev = StreamEvent(
            type="final",
            segment_index=self._seg_index,
            seq=self._seq,
            start=seg.start,
            end=seg.end,
            text=best,
            delta=delta,
            full_text=self._history_text,
            is_final=True,
        )
        self._seg_index += 1
        self._reset_utterance(cur_end)
        return ev

    def feed(self, pcm_bytes: bytes) -> Iterator[StreamEvent]:
        """向流式会话推入任意长度的 16kHz/16bit/mono PCM 字节块"""
        if not pcm_bytes:
            return
        self._pcm_buf.extend(pcm_bytes)
        self._total_pcm_bytes += len(pcm_bytes)

        while len(self._pcm_buf) >= self._chunk_bytes:
            chunk = bytes(self._pcm_buf[: self._chunk_bytes])
            del self._pcm_buf[: self._chunk_bytes]

            ev = self._send_chunk(chunk)
            if ev is not None:
                yield ev

            utt_dur = self._utt_pcm_bytes / 32000.0
            should_cut = (
                self._utt_done
                or (utt_dur >= self.min_utterance_sec and self._utt_silence_sec >= self.vad_silence_sec)
                or (utt_dur >= self.max_utterance_sec)
            )
            if should_cut:
                fin_ev = self._finalize_utterance()
                if fin_ev is not None:
                    yield fin_ev

    def flush(self) -> Iterator[StreamEvent]:
        """冲刷缓冲区剩余不足一个 chunk 的尾帧音频并完成最终定稿"""
        if self._pcm_buf:
            rem = bytes(self._pcm_buf)
            self._pcm_buf.clear()
            ev = self._send_chunk(rem)
            if ev is not None:
                yield ev

        if self._utt_pcm_bytes > 0:
            fin_ev = self._finalize_utterance()
            if fin_ev is not None:
                yield fin_ev

    @property
    def result(self) -> TranscriptResult:
        res = TranscriptResult(
            segments=list(self.segments),
            duration=round(self._total_pcm_bytes / 32000.0, 2),
            source=self.source_name,
            elapsed_sec=round(time.perf_counter() - self._t0, 2),
        )
        if self.organize and res.raw_text:
            res.organized_text = self.client.organize_text(res.raw_text)
            res.elapsed_sec = round(time.perf_counter() - self._t0, 2)
        return res

# -*- coding: utf-8 -*-
"""
baidu-asr 单元测试与端到端实测套件：
验证零第三方依赖百度二进制协议编解码、Opus restricted_lowdelay 32kbps CBR 80B 帧编码（修复 640B 步长）、
流式识别、向后兼容 `recognize` 接口与多线程并发性能。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from baidu_asr import (
    BaiduASR,
    BaiduOpusEncoder,
    compute_rms,
    encode_pcm_to_baidu_ws_frames,
    load_audio,
    opus_packets,
    recognize,
    split_pcm_by_silence,
    transcribe,
    transcribe_stream,
)
from baidu_asr.codec import (
    MSG_TYPE_FINAL,
    MSG_TYPE_READY_ACK,
    MSG_TYPE_SESSION_END,
    compute_delta,
    eof_frame,
    frame,
    param_frame,
    parse_baidu_message,
    record,
)


class TestBaiduASRUnit(unittest.TestCase):
    def test_codec_frames_and_records(self):
        pf = param_frame("test-sn-123")
        self.assertGreater(len(pf), 20)
        body_len = int.from_bytes(pf[:4], "little")
        self.assertEqual(body_len, len(pf) - 4)
        self.assertEqual(pf[4:8], bytes([3, 1, 0, 0]))

        rec = record(b"\x11" * 80)
        self.assertEqual(len(rec), 88)
        self.assertEqual(rec[:8], bytes.fromhex("0000005001000000"))

        ef = eof_frame()
        self.assertEqual(ef, bytes.fromhex("0400000003010003"))

        mtype, parsed = parse_baidu_message(bytes.fromhex("04000000030100f4"))
        self.assertEqual(mtype, MSG_TYPE_READY_ACK)
        self.assertIsNone(parsed)

        sample_f1 = frame(
            bytes([3, 1, 0, MSG_TYPE_FINAL])
            + json.dumps({"err_no": 0, "result": {"word": ["测试文本"], "confident": [99]}}).encode("utf-8")
        )
        mtype2, d2 = parse_baidu_message(sample_f1)
        self.assertEqual(mtype2, MSG_TYPE_FINAL)
        self.assertEqual(d2["result"]["word"][0], "测试文本")

    def test_opus_restricted_lowdelay_cbr80(self):
        # 1 秒 16kHz 16-bit mono PCM = 32000 字节 -> 精确 50 个 20ms 包，每包 CBR 80 字节
        pcm = b"\x10\x00" * 16000
        pkts = opus_packets(pcm)
        self.assertEqual(len(pkts), 50)
        for p in pkts:
            self.assertEqual(len(p), 80)

        ws_blob = encode_pcm_to_baidu_ws_frames(pcm, include_eof=True)
        self.assertGreater(len(ws_blob), 4000)

    def test_live_asr_and_streaming_and_concurrency(self):
        sample_wav = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "examples", "asr_example_zh.wav")
        )
        sample_wav2 = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "examples", "asr_example.wav")
        )
        if not os.path.isfile(sample_wav):
            self.skipTest("Sample wav not found")

        # 1. 测试向后兼容的 load_audio 与 recognize 异步接口
        pcm1 = load_audio(sample_wav, 16000)
        partials = []
        txt = asyncio.run(recognize(pcm1, on_partial=lambda t: partials.append(t)))
        self.assertIn("达摩院", txt)
        self.assertGreaterEqual(len(partials), 1)

        # 2. 测试流式识别 transcribe_stream
        events = list(transcribe_stream(sample_wav, chunk_sec=0.8))
        self.assertGreaterEqual(len(events), 1)
        self.assertTrue(events[-1].is_final)
        self.assertIn("达摩院", events[-1].text)

        # 3. 测试多线程并发长音频转写 + 可选豆包 LLM 整理
        with BaiduASR() as asr:
            pcm2 = asr.load_pcm16k(sample_wav2)
            silence_1s = b"\x00\x00" * 16000
            long_pcm = pcm1 + silence_1s + pcm2 + silence_1s + pcm1

            res = asr.transcribe(
                long_pcm,
                workers=4,
                min_segment_sec=4.0,
                max_segment_sec=7.0,
                organize=True,
            )
            print(
                f"\n[Baidu Benchmark] {res.duration:.1f}s 音频 ({len(res.segments)} 分片) "
                f"4线程转写+整理总耗时: {res.elapsed_sec:.2f}s ({res.duration/max(0.01, res.elapsed_sec):.1f}x 实时)"
            )
            print(f"[Baidu Benchmark] 百度 ASR 原文: {res.raw_text}")
            print(f"[Baidu Benchmark] 智能整理成稿: {res.organized_text}")
            self.assertIn("达摩院", res.raw_text)
            self.assertTrue(len(res.to_srt()) > 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# -*- coding: utf-8 -*-
"""
baidu_asr — 零依赖高性能百度输入法长语音与音视频转文字 SDK
"""

from .audio import (
    BaiduOpusEncoder,
    compute_rms,
    encode_pcm_to_baidu_ws_frames,
    iter_pcm16k_chunks,
    load_pcm16k,
    opus_packets,
    split_pcm_by_silence,
)
from .client import BaiduASR, load_audio, organize_text, recognize, transcribe, transcribe_stream
from .models import Segment, StreamEvent, TranscriptResult
from .stream import BaiduStreamSession

__version__ = "1.0.0"
__all__ = [
    "BaiduASR",
    "BaiduStreamSession",
    "transcribe",
    "transcribe_stream",
    "recognize",
    "load_audio",
    "organize_text",
    "Segment",
    "StreamEvent",
    "TranscriptResult",
    "BaiduOpusEncoder",
    "opus_packets",
    "encode_pcm_to_baidu_ws_frames",
    "compute_rms",
    "split_pcm_by_silence",
    "iter_pcm16k_chunks",
    "load_pcm16k",
]

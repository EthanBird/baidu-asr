# -*- coding: utf-8 -*-
"""
命令行工具入口 (`python -m baidu_asr` / `baidu-asr`)
"""

from __future__ import annotations

import argparse
import sys

from .client import BaiduASR
from .models import Segment


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        prog="baidu-asr",
        description="Baidu ASR — 零依赖高性能百度输入法长语音与音视频转文字工具",
    )
    parser.add_argument("input", nargs="+", help="输入音频或视频文件路径 (wav/mp3/m4a/mp4/m4s/flac...)")
    parser.add_argument("-o", "--output", help="输出文件路径 (可选，如 out.srt / out.md / out.txt)")
    parser.add_argument(
        "-f",
        "--format",
        choices=["txt", "srt", "vtt", "md", "json"],
        default=None,
        help="输出格式: txt | srt | vtt | md | json",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=8,
        help="并发识别线程数 (默认 8，仅在非 --stream 批量模式下生效)",
    )
    parser.add_argument("--stream", action="store_true", help="启用协议级真流式模式（实时显示增量识别）")
    parser.add_argument("--organize", action="store_true", help="转写后调用免鉴权豆包官方 LLM「智能整理」补齐标点并分段")
    parser.add_argument("--pid", type=int, default=597, help="百度输入法 product_id (默认 597)")
    parser.add_argument("-r", "--rate", type=int, default=16000, help="兼容原版参数：裸 PCM 采样率 (默认 16000)")
    parser.add_argument("--no-split", action="store_true", help="长音频不按静音分段")
    parser.add_argument("-q", "--quiet", action="store_true", help="静默模式，不打印中间进度")
    args = parser.parse_args()

    with BaiduASR() as asr:
        for src in args.input:
            if args.stream:
                with asr.create_stream(
                    chunk_sec=0.6,
                    pid=args.pid,
                    organize=args.organize,
                    source_name=src,
                ) as stream:
                    for pcm_chunk in asr.iter_pcm16k_chunks(src, chunk_bytes=int(0.6 * 32000)):
                        for ev in stream.feed(pcm_chunk):
                            if args.quiet:
                                continue
                            if ev.is_final:
                                print(
                                    f"\r[定稿 #{ev.segment_index:02d}] [{ev.timestamp}] {ev.text}"
                                    + " " * 15,
                                    flush=True,
                                )
                            else:
                                print(
                                    f"\r[流式 #{ev.segment_index:02d}] [{ev.timestamp}] {ev.text} ...",
                                    end="",
                                    flush=True,
                                )
                    for ev in stream.flush():
                        if not args.quiet and ev.is_final:
                            print(
                                f"\r[定稿 #{ev.segment_index:02d}] [{ev.timestamp}] {ev.text}"
                                + " " * 15,
                                flush=True,
                            )
                res = stream.result
            else:
                def _printer(seg: Segment):
                    if not args.quiet:
                        txt = seg.text if seg.text else "(静音/BGM)"
                        print(f"[{seg.index:02d}] [{seg.timestamp}] {txt}", flush=True)

                res = asr.transcribe(
                    src,
                    split=not args.no_split,
                    workers=args.workers,
                    pid=args.pid,
                    organize=args.organize,
                    on_segment=_printer,
                )

            if args.output and len(args.input) == 1:
                saved = res.save(args.output, fmt=args.format)
                if not args.quiet:
                    print(
                        f"\n[✓] 已保存至: {saved} (音频 {res.duration:.1f}s, "
                        f"耗时 {res.elapsed_sec:.2f}s, 加速比 {res.duration / max(0.001, res.elapsed_sec):.1f}x)"
                    )
            elif not args.stream or args.format:
                fmt = args.format or "txt"
                if not args.quiet:
                    print("=" * 60)
                if fmt == "srt":
                    print(res.to_srt())
                elif fmt == "vtt":
                    print(res.to_vtt())
                elif fmt == "json":
                    print(res.to_json())
                elif fmt == "md":
                    print(res.to_markdown())
                else:
                    print(res.text)


if __name__ == "__main__":
    main()

# baidu-asr — 零依赖高性能百度输入法长语音与音视频转文字 SDK

`baidu-asr` 是对百度输入法语音识别协议（`wss://vse.baidu.com/ws/asr`，`pid=597`）进行全面工程化重构、Bug 修复与性能突破后的 **零第三方 pip 依赖 (`dependencies = []`)** Python SDK。

---

## 一、 核心重构与原版脚本 (`01-baidu_asr.py`) 对比

| 优化维度 | 原版单脚本 (`01-baidu_asr.py`) | 重构版 `baidu-asr` SDK | 实际收益 |
| :--- | :--- | :--- | :--- |
| **Opus 切片步长减半与越界 Bug 修复** | `step = RATE // 50 = 320` 字节切片 `pcm[i:i+320]`，却传 `enc.encode(chunk, 320)`（320 samples = 640 字节）。导致 **每次越界多读 320B 内存 + 相邻两帧 50% 重叠**，把 5.56 秒音频强行变成 555 个包（`raf=1112`，膨胀为 11.12 秒），中间识别严重失真（误识为`“莱迪燕单模院”`） | 修正步长为精确的 **`640` 字节**（`320` samples = 20ms @ 16kHz） | 5.56 秒音频精确生成 **278 个包 (`raf=556`)**，彻底消除帧重叠与越界失真 |
| **消除三大硬编码延时瓶颈** | 1. 发送 `param_frame` 后盲等 `asyncio.sleep(0.2)`；<br>2. 每发 2 个包强制 `await asyncio.sleep(0.04)`；<br>3. 末尾循环靠 `timeout=3` 超时退出。<br>**导致识别 5.5 秒短音频耗时长达 `13.45s`（仅 `0.4x` 实时！）** | 1. 精确等待服务端 8 字节 `03 01 00 f4` (Ready ACK)；<br>2. 预打包全部双 Record 帧并通过 **单次 `sendall()`** 写出（零 `sleep`）；<br>3. 收到服务端 `03 01 00 f3` (Session End ACK) 立即返回 | **5.5 秒音频识别耗时从 `13.45s` 降至 `~0.9s`（提速 15 倍）**；8 线程并发跑 120s 音频仅需 **`2.60s`（提速 100 倍！）** |
| **彻底零第三方依赖 (`dependencies = []`)** | 依赖 `websockets` 与已停止维护的 `opuslib` | 原生标准库 `ssl` + `socket` RFC 6455 `BaiduWSSConnection` + 直接 `ctypes` 绑定内置 `_bin/opus.dll` | **零 pip 依赖开箱即用** |
| **任意音视频格式与智能标点成稿** | 仅支持 `.wav` 或裸 `.pcm`；且 `pid=597` 原生返回无标点文本 | 内置 `ffmpeg` 流式管道支持 `.mp4`/`.m4s`/`.mp3`/`.m4a`/`.flac` 等任意格式；内置可选免鉴权豆包官方 LLM「智能整理」(`organize=True`) | 自动为百度无标点原文补齐标点、剔除水词并自然分段 |

---

## 二、 项目目录结构

```text
baidu-asr/
├── pyproject.toml                # 零依赖标准打包配置 (dependencies = [])
├── README.md                     # 架构说明与使用文档
├── demo.ipynb                    # 含真实执行输出的 9 种全接口调用演示 Notebook
├── benchmarks/                   # 长视频实测转写成果与 SRT 字幕
├── examples/
│   ├── asr_example_zh.wav        # 中文示例语音
│   └── asr_example.wav           # 短语音示例
├── src/
│   └── baidu_asr/
│       ├── __init__.py           # 顶层公开 API 导出 (BaiduASR, BaiduStreamSession, transcribe, recognize, ...)
│       ├── models.py             # 数据模型 (Segment, StreamEvent, TranscriptResult)
│       ├── codec.py              # 百度 4B 小端帧/双 Record 编解码 + RFC 6455 帧打包 + 增量文本差分
│       ├── audio.py              # 直接 ctypes libopus 绑定 (restricted_lowdelay 32kbps CBR, 修复 640B 步长) + VAD 切分
│       ├── transport.py          # 原生 ssl+socket BaiduWSSConnection (0xf4 Ready ACK + 单次 sendall + 0xf3 即时结束)
│       ├── stream.py             # 协议级真流式推流会话 BaiduStreamSession
│       ├── client.py             # 高级客户端 BaiduASR (8线程并发 + 同步/异步流式 + 兼容原版 recognize/load_audio)
│       ├── cli.py                # CLI 命令行入口 (支持 -w 并发数、--stream、--organize、-f txt/srt/vtt/md/json)
│       ├── __main__.py           # 支持 python -m baidu_asr
│       └── _bin/
│           └── opus.dll          # 内置 64 位 libopus 动态库，开箱即用
└── tests/
    └── test_baidu_asr.py         # 单元测试 + 端到端流式与多线程并发集成测试套件
```

---

## 三、 快速上手

### 1. 一行代码极速转写 + 可选智能标点整理 (`transcribe`)

```python
from baidu_asr import transcribe

res = transcribe(
    "examples/asr_example_zh.wav",
    workers=8,
    organize=True,  # 可选：自动调用免鉴权官方 LLM 补齐标点并分段
)

print("整理成稿:", res.text)
print("百度原文:", res.raw_text)
print("分段明细:", res.segments)
res.save("output.srt")  # 导出 SRT 字幕
res.save("output.md")   # 导出 Markdown 报告
```

### 2. 向后兼容原版 `recognize` 异步接口与 `load_audio`

```python
import asyncio
from baidu_asr import recognize, load_audio

pcm = load_audio("examples/asr_example_zh.wav", 16000)
text = asyncio.run(recognize(pcm, on_partial=lambda t: print("partial:", t)))
print("final:", text)
```

### 3. 生成器流式接口 `transcribe_stream` 与实时推流 `create_stream`

```python
from baidu_asr import transcribe_stream

for ev in transcribe_stream("examples/asr_example_zh.wav", chunk_sec=0.6):
    print(f"[{ev.type.upper():7s}] [{ev.timestamp}] {ev.text} (+{ev.delta})")
```

### 4. CLI 命令行使用

```powershell
# 8 线程并发转写长音视频并启用智能标点整理，导出 Markdown 报告
python -m baidu_asr input.mp4 -w 8 --organize -f md -o report.md

# 导出 SRT 字幕
python -m baidu_asr input.mp4 -w 8 -f srt -o subtitle.srt

# 协议级真流式实时转写模式
python -m baidu_asr input.mp4 --stream
```

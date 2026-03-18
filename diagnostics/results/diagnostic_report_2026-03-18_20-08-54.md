# Voice Assistant — Diagnostic Report

> Generated: 2026-03-18_20-08-54  
> Git commit: `unknown`

## Health Summary

- ✅ GPU available
- ✅ Groq API reachable
- ✅ API key valid
- ✅ STT latency good
- ✅ LLM TTFT good
- ✅ TTS latency good
- ✅ Echo decays cleanly
- ✅ VAD clean on silence
- ✅ System prompt compliant


## Hardware

### CPU
- **Model:** Architecture:                            x86_64
- **Cores:** 6p / 12t
- **Max frequency:** 5600.0 MHz
- **Current usage:** 1.5 %

### RAM
- **Total:** 32.95 GB
- **Used:** 3.77 GB
- **Available:** 29.19 GB
- **Usage:** 11.4 %

### GPU
- **Name:** NVIDIA GeForce RTX 2070 SUPER
- **VRAM total:** 8.16 GB
- **VRAM used:** 0.42 GB
- **VRAM usage:** 5.1 %
- **Compute cap:** 7.5
- **CUDA version:** 12.8
- **Driver version:** 590.48.01

### Disk
- **Read speed:** 3073.3 MB/s
- **Write speed:** 4162.5 MB/s
- **Free space:** 359.8 GB

### Audio System
- **Server:** PulseAudio (on PipeWire 1.4.7)
- **Version:** 15.0.0
- **Mic device:** alsa_input.pci-0000_00_1f.3.analog-stereo
- **Mic volume:** Volume: front-left: 31257 /  48% / -19.29 dB,   front-right: 31257 /  48% / -19.29 dB

### Library Versions
| Library        | Version                 |
| -------------- | ----------------------- |
| python         | 3.13.7                  |
| os             | Linux 6.17.0-19-generic |
| torch          | 2.10.0+cu128            |
| cuda           | 12.8                    |
| groq           | 1.1.1                   |
| faster_whisper | 1.2.1                   |
| kokoro_onnx    | 0.5.0                   |
| silero_vad     | 6.2.1                   |
| pyaudio        | 0.2.14                  |
| sounddevice    | 0.5.5                   |
| onnxruntime    | not installed           |
| numpy          | 2.4.3                   |

## Audio

### Device Latency
- **Input device:** default
- **Input latency:** 8.68 ms
- **Output device:** default
- **Output latency:** 8.68 ms

### Mic Noise Floor
- **Mean RMS:** 0.60071
- **Max RMS:** 1.0
- **Std RMS:** 0.3852
- **Recommended `MIN_AUDIO_ENERGY`:** 1.7563

### Echo Decay
- **Playback duration:** 6.06 s
- **Peak RMS after playback:** 0.0884
- **Mean RMS after playback:** 0.0189
- **Echo decay time:** 0.04 s
- **Echo fully decayed:** ✅

> **Recommendations:**
> - `BUFFER_MUTE_GUARD = 0.5` seconds
> - `POST_SPEECH_MUTE = 0.3` seconds

### VAD False Positive Rate (on silence)
- **Status:** ✅ good
- **False positives:** 0 / 156 chunks
- **False positive rate:** 0.0 %
- **Mean VAD prob:** 0.0016
- **Max VAD prob:** 0.0121
- **VAD threshold:** 0.5

### VAD Detection Latency
- **Detection latency:** not detected ms
- **Chunk duration:** 32.0 ms

## Network
- **Groq reachable:** ✅

### DNS
- **Host:** api.groq.com
- **Resolved IP:** 104.18.38.236
- **Latency:** 17.02 ms

### TLS Handshake
- **Latency:** 111.38 ms

### HTTPS Ping to Groq API
| Metric  | Value     |
| ------- | --------- |
| Samples | 10        |
| Min     | 268.41 ms |
| Mean    | 272.85 ms |
| Max     | 289.06 ms |
| Jitter  | 5.63 ms   |

### API Key
- **Valid:** ✅
- **Models available:** 18

### Bandwidth
- **Upload speed:** 2.73 Mbps
- **Est. 1s WAV upload:** 91.4 ms
- **Est. 3s WAV upload:** 274.3 ms
- **Est. 5s WAV upload:** 457.1 ms
- **Download speed:** 32.62 Mbps

## STT (Groq Whisper)
- **Status:** ✅ good
- **Model:** whisper-large-v3-turbo
- **Language:** en
- **Avg latency:** 0.215 s
- **Model available:** ✅

### Latency by Audio Length
| Length | WAV size | Min    | Mean   | Max    | Std    | RTF    |
| ------ | -------- | ------ | ------ | ------ | ------ | ------ |
| 1s     | 31.3 KB  | 0.204s | 0.219s | 0.245s | 0.018s | 0.219x |
| 3s     | 93.8 KB  | 0.205s | 0.216s | 0.228s | 0.009s | 0.072x |
| 5s     | 156.3 KB | 0.203s | 0.211s | 0.221s | 0.008s | 0.042x |

### Accuracy (Kokoro → Groq round trip)
- **Overall accuracy:** 93.3 %

- **Expected:** `Hello, how are you today?`  
  **Got:** `Hello, how are you today?`  
  Accuracy: 100.0%  |  Exact match: ✅  |  Latency: 0.283s

- **Expected:** `The weather is nice outside.`  
  **Got:** `The weather is nice outside.`  
  Accuracy: 100.0%  |  Exact match: ✅  |  Latency: 0.283s

- **Expected:** `My name is John and I am thirty years old.`  
  **Got:** `My name is Jan and I am 30 years old.`  
  Accuracy: 80.0%  |  Exact match: ❌  |  Latency: 0.334s

## LLM (Groq)
- **Status:** ✅ good
- **Model:** meta-llama/llama-4-scout-17b-16e-instruct
- **Short TTFT:** 0.252 s
- **Max tokens:** 120
- **Temperature:** 0.4

### Latency by Prompt Complexity
| Complexity | Words | TTFT min | TTFT mean | TTFT max | Throughput  | Avg tokens |
| ---------- | ----- | -------- | --------- | -------- | ----------- | ---------- |
| short      | 1     | 0.218s   | 0.252s    | 0.269s   | 405.6 tok/s | 18.7       |
| medium     | 12    | 0.1s     | 0.276s    | 0.449s   | 414.7 tok/s | 46.3       |
| long       | 20    | 0.099s   | 0.102s    | 0.104s   | 427.7 tok/s | 107.7      |

### Conversation History Overhead
| Turns | ~Tokens | Mean TTFT | Min TTFT | Max TTFT |
| ----- | ------- | --------- | -------- | -------- |
| 0     | 0       | 0.128s    | 0.1s     | 0.181s   |
| 5     | 75      | 0.148s    | 0.105s   | 0.203s   |
| 10    | 150     | 0.11s     | 0.109s   | 0.112s   |

> **History overhead:** -1.80ms per additional turn

### System Prompt Compliance
- **Compliance score:** 100.0 %
- **Word count:** 56
- **Sentence count:** 2
- ✅ no bullet points
- ✅ no markdown
- ✅ within 2 sentences
- ✅ concise under 80 words

**Sample response:**
> Quantum computing is a type of computing that uses the principles of quantum mechanics to perform calculations and operations on data. It's like a super-powerful computer that can solve certain problems way faster than a regular computer, by using quantum bits or qubits that can exist in many states...

## TTS (Kokoro)
- **Status:** ✅ good
- **Voice:** bf_emma
- **Speed:** 1.15
- **Medium phrase latency:** 0.44 s
- **Medium phrase RTF:** 0.217

### Model Files
- **ONNX model:** kokoro-v0_19.onnx
- **ONNX size:** 325.5 MB
- **ONNXRuntime version:** 1.24.3
- **Providers:** AzureExecutionProvider, CPUExecutionProvider

### Cold Start vs Warmed Up
- **Cold start latency:** 0.257 s
- **Warmed up latency:** 0.234 s
- **Overhead:** 0.023 s
- **Cold RTF:** 0.228
- **Warm RTF:** 0.207

### Synthesis Latency by Phrase Length
| Length | Chars | Min    | Mean   | Max    | Audio dur | RTF    | Chars/s  |
| ------ | ----- | ------ | ------ | ------ | --------- | ------ | -------- |
| short  | 13    | 0.234s | 0.235s | 0.236s | 1.131s    | 0.208x | 55.3 c/s |
| medium | 39    | 0.438s | 0.44s  | 0.442s | 2.027s    | 0.217x | 88.6 c/s |
| long   | 111   | 1.727s | 1.737s | 1.743s | 8.619s    | 0.202x | 63.9 c/s |

### Word Boundary Split Test
- **Original chars:** 126
- **Split at char:** 116
- **Part 1 chars:** 116
- **Part 2 chars:** 9
- **Clean split:** ✅
- **Part 1 latency:** 1.151 s
- **Part 2 latency:** 0.212 s

### GPU During Synthesis
- **VRAM before:** 418.0 MB
- **VRAM after:** 418.0 MB
- **VRAM delta:** 0.0 MB

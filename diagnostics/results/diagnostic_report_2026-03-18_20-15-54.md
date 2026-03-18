# Voice Assistant — Diagnostic Report

> Generated: 2026-03-18_20-15-54  
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
- **Current usage:** 1.0 %

### RAM
- **Total:** 32.95 GB
- **Used:** 3.78 GB
- **Available:** 29.18 GB
- **Usage:** 11.5 %

### GPU
- **Name:** NVIDIA GeForce RTX 2070 SUPER
- **VRAM total:** 8.16 GB
- **VRAM used:** 0.41 GB
- **VRAM usage:** 5.1 %
- **Compute cap:** 7.5
- **CUDA version:** 12.8
- **Driver version:** 590.48.01

### Disk
- **Read speed:** 3099.3 MB/s
- **Write speed:** 4157.5 MB/s
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
- **Mean RMS:** 0.50339
- **Max RMS:** 0.99997
- **Std RMS:** 0.40007
- **Recommended `MIN_AUDIO_ENERGY`:** 1.7036

### Echo Decay
- **Playback duration:** 6.06 s
- **Peak RMS after playback:** 0.0183
- **Mean RMS after playback:** 0.0112
- **Echo decay time:** 0.04 s
- **Echo fully decayed:** ✅

> **Recommendations:**
> - `BUFFER_MUTE_GUARD = 0.5` seconds
> - `POST_SPEECH_MUTE = 0.3` seconds

### VAD False Positive Rate (on silence)
- **Status:** ✅ good
- **False positives:** 0 / 156 chunks
- **False positive rate:** 0.0 %
- **Mean VAD prob:** 0.002
- **Max VAD prob:** 0.0087
- **VAD threshold:** 0.5

### VAD Detection Latency
- **Detection latency:** not detected ms
- **Chunk duration:** 32.0 ms

## Network
- **Groq reachable:** ✅

### DNS
- **Host:** api.groq.com
- **Resolved IP:** 172.64.149.20
- **Latency:** 16.36 ms

### TLS Handshake
- **Latency:** 71.09 ms

### HTTPS Ping to Groq API
| Metric  | Value     |
| ------- | --------- |
| Samples | 10        |
| Min     | 254.21 ms |
| Mean    | 256.24 ms |
| Max     | 258.41 ms |
| Jitter  | 1.37 ms   |

### API Key
- **Valid:** ✅
- **Models available:** 18

### Bandwidth
- **Upload speed:** 2.87 Mbps
- **Est. 1s WAV upload:** 87.1 ms
- **Est. 3s WAV upload:** 261.2 ms
- **Est. 5s WAV upload:** 435.4 ms
- **Download speed:** 40.04 Mbps

## STT (Groq Whisper)
- **Status:** ✅ good
- **Model:** whisper-large-v3-turbo
- **Language:** en
- **Avg latency:** 0.188 s
- **Model available:** ✅

### Latency by Audio Length
| Length | WAV size | Min    | Mean   | Max    | Std    | RTF    |
| ------ | -------- | ------ | ------ | ------ | ------ | ------ |
| 1s     | 31.3 KB  | 0.177s | 0.187s | 0.196s | 0.008s | 0.187x |
| 3s     | 93.8 KB  | 0.182s | 0.186s | 0.193s | 0.005s | 0.062x |
| 5s     | 156.3 KB | 0.191s | 0.192s | 0.193s | 0.001s | 0.038x |

### Accuracy (Kokoro → Groq round trip)
- **Overall accuracy:** 93.3 %

- **Expected:** `Hello, how are you today?`  
  **Got:** `Hello, how are you today?`  
  Accuracy: 100.0%  |  Exact match: ✅  |  Latency: 0.243s

- **Expected:** `The weather is nice outside.`  
  **Got:** `The weather is nice outside.`  
  Accuracy: 100.0%  |  Exact match: ✅  |  Latency: 0.237s

- **Expected:** `My name is John and I am thirty years old.`  
  **Got:** `My name is Jan and I am 30 years old.`  
  Accuracy: 80.0%  |  Exact match: ❌  |  Latency: 0.262s

## LLM (Groq)
- **Status:** ✅ good
- **Model:** meta-llama/llama-4-scout-17b-16e-instruct
- **Short TTFT:** 0.132 s
- **Max tokens:** 120
- **Temperature:** 0.4

### Latency by Prompt Complexity
| Complexity | Words | TTFT min | TTFT mean | TTFT max | Throughput  | Avg tokens |
| ---------- | ----- | -------- | --------- | -------- | ----------- | ---------- |
| short      | 1     | 0.094s   | 0.132s    | 0.2s     | 454.6 tok/s | 17.0       |
| medium     | 12    | 0.093s   | 0.112s    | 0.129s   | 420.4 tok/s | 49.0       |
| long       | 20    | 0.09s    | 0.111s    | 0.15s    | 435.4 tok/s | 95.7       |

### Conversation History Overhead
| Turns | ~Tokens | Mean TTFT | Min TTFT | Max TTFT |
| ----- | ------- | --------- | -------- | -------- |
| 0     | 0       | 0.147s    | 0.092s   | 0.178s   |
| 5     | 75      | 0.097s    | 0.094s   | 0.1s     |
| 10    | 150     | 0.103s    | 0.1s     | 0.106s   |

> **History overhead:** -4.40ms per additional turn

### System Prompt Compliance
- **Compliance score:** 75.0 %
- **Word count:** 81
- **Sentence count:** 2
- ✅ no bullet points
- ✅ no markdown
- ✅ within 2 sentences
- ❌ concise under 80 words

**Sample response:**
> Quantum computing is a new way of processing information that's different from classical computers, it uses the principles of quantum mechanics to perform calculations, like superpositions and entanglements, to solve problems that are too complex for regular computers. For example, imagine you have ...

## TTS (Kokoro)
- **Status:** ✅ good
- **Voice:** bf_emma
- **Speed:** 1.15
- **Medium phrase latency:** 0.438 s
- **Medium phrase RTF:** 0.216

### Model Files
- **ONNX model:** kokoro-v0_19.onnx
- **ONNX size:** 325.5 MB
- **ONNXRuntime version:** 1.24.3
- **Providers:** AzureExecutionProvider, CPUExecutionProvider

### Cold Start vs Warmed Up
- **Cold start latency:** 0.263 s
- **Warmed up latency:** 0.368 s
- **Overhead:** -0.105 s
- **Cold RTF:** 0.233
- **Warm RTF:** 0.326

### Synthesis Latency by Phrase Length
| Length | Chars | Min    | Mean   | Max    | Audio dur | RTF    | Chars/s  |
| ------ | ----- | ------ | ------ | ------ | --------- | ------ | -------- |
| short  | 13    | 0.235s | 0.236s | 0.237s | 1.131s    | 0.208x | 55.2 c/s |
| medium | 39    | 0.432s | 0.438s | 0.443s | 2.027s    | 0.216x | 89.1 c/s |
| long   | 111   | 1.724s | 1.74s  | 1.752s | 8.619s    | 0.202x | 63.8 c/s |

### Word Boundary Split Test
- **Original chars:** 126
- **Split at char:** 116
- **Part 1 chars:** 116
- **Part 2 chars:** 9
- **Clean split:** ✅
- **Part 1 latency:** 1.148 s
- **Part 2 latency:** 0.213 s

### GPU During Synthesis
- **VRAM before:** 412.2 MB
- **VRAM after:** 412.2 MB
- **VRAM delta:** 0.0 MB

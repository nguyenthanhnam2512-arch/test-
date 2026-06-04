# Music Source Separation + Speech Enhancement — Experiment

## Mục tiêu

Đánh giá pipeline **tách giọng nói** từ audio YouTube thô (có nhạc nền, tạp âm) và **tăng cường chất lượng** để thu được giọng nói sạch nhất.

```
raw YouTube audio (.webm)
    ↓
[Convert] ffmpeg → WAV (44.1kHz stereo)
    ↓
[Step 1] Music Source Separation
    ├── BS-Roformer-Viperx-1297       (vocal SDR ≈ 11.77 dB)
    └── MelBand-Roformer-Vocal (Kim)  (vocal SDR ≈ 12.60 dB)
    ↓
[Step 2] Speech Enhancement
    └── DeepFilterNet 3
    ↓
[Eval] DNSMOS P.835 (SIG, BAK, OVRL)
```

## Models so sánh

| Model | Architecture | Vocal SDR | Ghi chú |
|-------|-------------|-----------|---------|
| BS-Roformer-Viperx-1297 | Band-Split RoFormer | 11.77 dB | State-of-the-art v1 |
| MelBand-Roformer-Vocal | Mel-Band RoFormer (Kim FT) | 12.60 dB | State-of-the-art v2 |

## Cấu trúc thư mục

```
separation_experiment/
├── run_experiment.py       # Script chính
├── models/                 # Model checkpoints (tự động tải khi chạy)
├── audio_raw_wav/          # WAV đã convert từ webm
├── audio_separated/
│   ├── bs_roformer_1297/   # Vocals tách bởi BS-Roformer
│   └── melband_roformer_vocal/ # Vocals tách bởi MelBand-Roformer
├── audio_enhanced/
│   ├── bs_roformer_1297/   # Vocals → DeepFilterNet3
│   └── melband_roformer_vocal/
└── results.json            # Kết quả DNSMOS tổng hợp
```

## Chạy thực nghiệm

```bash
cd "E:\AI thuc chien\AI voice\Test Speech\separation_experiment"
python run_experiment.py
```

**Lưu ý:** Lần đầu chạy, script sẽ tự tải model checkpoint (~200–400MB mỗi model).

## Dataset

- `audio/khanhvy_vlog/` – 23 video YouTube (vlog)
- `audio/khanhvy_talkshow/` – 37 video YouTube (talk show)
- `audio/beyondlimits_ngalevi/` – 29 video YouTube (interview/podcast)

Thực nghiệm lấy **3 file đầu mỗi category** (tổng 9 files) để đánh giá nhanh.

## Metrics

- **DNSMOS OVRL** – Overall MOS (thang 1–5), không cần reference audio
- **DNSMOS SIG** – Speech quality signal score
- **DNSMOS BAK** – Background noise quality score
- **latency_ms** – Độ trễ xử lý local từng bước (ms)
- **RTF** – Real-Time Factor (< 1.0 = xử lý nhanh hơn real-time)
- **SNR** – Signal-to-Noise Ratio ước tính (dB)

## Kết quả output

- `results.json` — DNSMOS / SNR + latency (ms)
- `latency_results.json` — độ trễ chi tiết MSS / enhance / pipeline
- `timings_cache.json` — cache thời gian MSS thực đo

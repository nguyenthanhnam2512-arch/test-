# Music Source Separation + Speech Enhancement — Experiment

> **Cập nhật 2026-06-08:** Thêm Voice Distortion metrics (F0-RMSE, MCD, EPR, VDS), HTML Listening Test, và so sánh với SIDON (arXiv:2509.17052v3).

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
[Eval] DNSMOS P.835 + Voice Distortion Metrics + Manual Listening Test
```

## Models so sánh

| Model | Architecture | Vocal SDR | Ghi chú |
|-------|-------------|-----------|---------|
| BS-Roformer-Viperx-1297 | Band-Split RoFormer | 11.77 dB | State-of-the-art v1 |
| MelBand-Roformer-Vocal | Mel-Band RoFormer (Kim FT) | 12.60 dB | State-of-the-art v2 |
| DeepFilterNet 3 | GRU + ERB filterbank | N/A | Speech Enhancement bước 2 |
| **SIDON** (ref) | w2v-BERT 2.0 + HiFi-GAN | N/A | [arXiv:2509.17052](https://arxiv.org/abs/2509.17052) — so sánh từ paper |

## Tại sao DNSMOS chưa đủ?

DNSMOS P.835 đo **noise suppression** nhưng **không phát hiện méo giọng**. Một model có thể đạt DNSMOS cao nhưng vẫn:
- Làm lệch pitch (F0-RMSE cao)
- Thay đổi âm sắc/timbre (MCD cao)
- Làm mất năng lượng giọng (EPR thấp)
- Robot hóa giọng (SFR tăng)

→ Cần **Voice Distortion metrics** bổ sung + **manual listening test**.

## Metrics đánh giá

| Metric | Mô tả | Ngưỡng tốt |
|--------|-------|------------|
| DNSMOS OVRL | Overall MOS (1–5) — noise suppression | > 3.5 |
| F0-RMSE (Hz) | Lệch pitch sep vs enhanced | < 15 Hz |
| MCD (dB) | Mel Cepstral Distortion (librosa scale) | < 20 dB |
| EPR | Energy Preservation Ratio | > 0.7 |
| SFR | Spectral Flatness Ratio | < 1.3 |
| VDS (0–10) | Voice Distortion Score tổng hợp | < 2.5 |
| RTF | Real-Time Factor | < 1.0 |

## Kết quả tóm tắt

| Pipeline | DNSMOS OVRL TB | VDS TB | F0-RMSE TB (Hz) | RTF CPU |
|----------|---------------|--------|-----------------|---------|
| BS-Roformer + DFN3 | **3.37** | ~3.2 | ~5.8 | 9.4× |
| **Mel-Band + DFN3** | **3.44** | ~3.0 | ~4.1 | 5.7× |
| SIDON (paper, English) | 3.31 | N/A | N/A | 0.002× GPU |

**Kết luận:** Pipeline tốt nhất cho YouTube tiếng Việt: **Mel-Band-Roformer + DeepFilterNet 3**.

## Cấu trúc thư mục

```
separation_experiment/
├── run_experiment.py           # Script chính: MSS + Enhancement + DNSMOS
├── perceptual_eval.py          # NEW: Voice Distortion metrics (F0, MCD, EPR, VDS)
├── generate_report.py          # Tạo báo cáo Word (.docx)
├── listening_test.html         # NEW: HTML listening test với manual scoring
├── results.json                # Kết quả DNSMOS tổng hợp
├── perceptual_results.json     # NEW: Kết quả Voice Distortion
├── latency_results.json        # Độ trễ chi tiết từng bước
├── models/                     # Model checkpoints
├── audio_raw_wav/              # WAV đã convert từ webm
├── audio_separated/
│   ├── bs_roformer_1297/       # Vocals tách bởi BS-Roformer
│   └── melband_roformer_vocal/ # Vocals tách bởi MelBand-Roformer
└── audio_enhanced/
    ├── bs_roformer_1297/       # Vocals → DeepFilterNet3
    └── melband_roformer_vocal/
```

## Chạy thực nghiệm

```bash
cd "Test Speech/separation_experiment"

# Bước 1: MSS + Enhancement + DNSMOS
python run_experiment.py

# Bước 2: Voice Distortion metrics
python perceptual_eval.py

# Bước 3: Tạo báo cáo Word
python generate_report.py

# Bước 4: Manual listening test → mở file trong browser
# listening_test.html
```

## Dataset

- `audio/khanhvy_vlog/` – 23 video YouTube (vlog, nhạc nền liên tục)
- `audio/khanhvy_talkshow/` – 37 video YouTube (talk show, nhạc nặng)
- `audio/beyondlimits_ngalevi/` – 29 video YouTube (interview/podcast, ít nhạc)

Thực nghiệm lấy **3 file đầu mỗi category** (tổng 9 files × 60s).

## So sánh với SIDON (arXiv:2509.17052v3)

| Tiêu chí | Pipeline chúng ta | SIDON |
|----------|------------------|-------|
| DNSMOS OVRL | **3.37–3.44** (ours) | 3.31 (English) |
| SpkSim | **~0.93–0.94** | 0.891 (vocoder resynthesis) |
| RTF | 5.7–9.4× CPU | **0.002× GPU** |
| GPU required | Không | Có (flash-attn) |
| Xử lý nhạc nền | Có (MSS step riêng) | Không (end-to-end) |
| Multilingual | Không (pretrained) | **Có (104 ngôn ngữ)** |
| Dereverberation | Không | **Có** |

→ **Pipeline ours tốt hơn** về speaker identity và không cần GPU. **SIDON tốt hơn** về tốc độ và scale.

## Outputs

| File | Mô tả |
|------|-------|
| `results.json` | DNSMOS / SNR + latency |
| `perceptual_results.json` | F0-RMSE, MCD, EPR, SFR, VDS |
| `latency_results.json` | Độ trễ chi tiết MSS / enhance / pipeline |
| `listening_test.html` | HTML listening test với audio player + manual scoring |
| `../../MSS_SpeechEnhancement_Report.docx` | Báo cáo Word đầy đủ |

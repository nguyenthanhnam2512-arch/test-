# Music Source Separation + Speech Enhancement — Thực nghiệm

Pipeline tách giọng nói từ audio YouTube thô (nhạc nền, tạp âm) và tăng cường chất lượng giọng.

```
YouTube .webm  →  WAV  →  MSS (Roformer)  →  DeepFilterNet 3  →  Clean voice
```

## Models so sánh

| Model | Mục đích |
|-------|----------|
| BS-Roformer-Viperx-1297 | Tách vocal khỏi nhạc nền |
| MelBand-Roformer-Vocal (Kim) | Tách vocal khỏi nhạc nền |
| DeepFilterNet 3 | Speech enhancement (bước 2) |

## Chạy thực nghiệm

```bash
cd "Test Speech/separation_experiment"
python run_experiment.py
```

Tạo báo cáo Word:

```bash
python generate_report.py
```

## Dataset

Đặt file `audio.zip` vào thư mục `Test Speech/` (không có trong repo — file quá lớn).  
Script tự giải nén, convert webm → WAV 60s, chạy 2 model MSS rồi DeepFilterNet3, đánh giá bằng DNSMOS P.835.

## Kết quả

- `results.json` — metrics DNSMOS / SNR
- `MSS_SpeechEnhancement_Report.docx` — báo cáo chi tiết (ở thư mục gốc repo)

## Yêu cầu

- Python 3.11+
- FFmpeg
- `pip install audio-separator deepfilternet torch torchaudio soundfile librosa onnxruntime python-docx numpy`

Model checkpoint (~1.5 GB) tự tải lần đầu chạy qua `audio-separator`.

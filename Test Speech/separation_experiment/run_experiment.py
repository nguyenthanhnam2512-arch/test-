# -*- coding: utf-8 -*-
"""
Thực nghiệm Music Source Separation (MSS) + Speech Enhancement Pipeline
========================================================================
Pipeline:
  raw YouTube audio (.webm)
      → [Step 1] Convert webm → WAV
      → [Step 2] Music Source Separation  (BS-Roformer-Viperx-1297 & MelBand-Roformer-Vocal)
      → [Step 3] Speech Enhancement       (DeepFilterNet 3)
      → [Step 4] DNSMOS P.835 evaluation  (SIG, BAK, OVRL)
      → [Step 5] Báo cáo so sánh

Models so sánh:
  - BS-Roformer-Viperx-1297        (Roformer BSS, vocal SDR ≈ 11.77 dB)
  - MelBand-Roformer-Vocal (Kim)   (Mel-Band Roformer, vocal SDR ≈ 12.60 dB)
"""

import sys, os, io, time, json, subprocess, zipfile, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import soundfile as sf

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
AUDIO_ZIP    = os.path.join(SCRIPT_DIR, "..", "audio.zip")
RAW_WAV_DIR  = os.path.join(SCRIPT_DIR, "audio_raw_wav")
SEP_OUT_DIR  = os.path.join(SCRIPT_DIR, "audio_separated")
ENH_OUT_DIR  = os.path.join(SCRIPT_DIR, "audio_enhanced")
RESULTS_JSON = os.path.join(SCRIPT_DIR, "results.json")
DNSMOS_ONNX  = os.path.join(SCRIPT_DIR, "..", "..", "denoise_experiment", "sig_bak_ovr.onnx")
MODEL_DIR    = os.path.join(SCRIPT_DIR, "models")

# ── Số file lấy mẫu từ mỗi category ──────────────────────────────
N_SAMPLES_PER_CATEGORY = 3

# ── Thời lượng trim (giây) — None = không trim ────────────────────
CLIP_DURATION_SEC = 60   # lấy 60s từ giữa video (bỏ 30s đầu)

# ── Tên model trong audio-separator ───────────────────────────────
SEPARATION_MODELS = {
    "bs_roformer_1297": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "melband_roformer_vocal": "vocals_mel_band_roformer.ckpt",
}

# ═══════════════════════════════════════════════════════════════════
# BƯỚC 0: Giải nén & convert webm → WAV
# ═══════════════════════════════════════════════════════════════════

def extract_and_convert(n_per_cat=N_SAMPLES_PER_CATEGORY):
    """Giải nén audio.zip, lấy N file đầu mỗi category, convert sang WAV 44100Hz stereo."""
    os.makedirs(RAW_WAV_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("BƯỚC 0: Giải nén audio.zip và convert → WAV")
    print("="*60)

    with zipfile.ZipFile(AUDIO_ZIP, "r") as zf:
        all_entries = [e for e in zf.namelist() if e.endswith(".webm")]
        # Nhóm theo category
        categories = {}
        for entry in all_entries:
            parts = entry.split("/")
            if len(parts) >= 3:
                cat = parts[1]
                categories.setdefault(cat, []).append(entry)

        print(f"  Categories: {list(categories.keys())}")
        selected = []
        for cat, files in categories.items():
            chosen = sorted(files)[:n_per_cat]
            selected.extend(chosen)
            print(f"  [{cat}] chọn {len(chosen)}/{len(files)} files")

        wav_files = []
        for entry in selected:
            name_no_ext = os.path.splitext(os.path.basename(entry))[0]
            cat = entry.split("/")[1]
            out_name = f"{cat}__{name_no_ext}.wav"
            out_path = os.path.join(RAW_WAV_DIR, out_name)

            if os.path.exists(out_path):
                print(f"  [SKIP] {out_name} đã có sẵn")
                wav_files.append(out_path)
                continue

            # Giải nén webm tạm thời
            webm_tmp = os.path.join(RAW_WAV_DIR, f"_tmp_{name_no_ext}.webm")
            with zf.open(entry) as src, open(webm_tmp, "wb") as dst:
                dst.write(src.read())

            # ffmpeg convert: 44100 Hz, stereo, PCM 16-bit
            # Lấy đoạn giữa video (bỏ 30s đầu) để tránh intro/jingle
            cmd = ["ffmpeg", "-y"]
            if CLIP_DURATION_SEC:
                cmd += ["-ss", "30", "-t", str(CLIP_DURATION_SEC)]
            cmd += [
                "-i", webm_tmp,
                "-ar", "44100", "-ac", "2", "-sample_fmt", "s16",
                out_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            os.remove(webm_tmp)

            if result.returncode == 0:
                size_kb = os.path.getsize(out_path) / 1024
                print(f"  ✓ {out_name} ({size_kb:.0f} KB)")
                wav_files.append(out_path)
            else:
                print(f"  ✗ FAIL: {out_name}")
                print(f"    {result.stderr[-300:]}")

    return wav_files


# ═══════════════════════════════════════════════════════════════════
# DNSMOS P.835 (no-reference)
# ═══════════════════════════════════════════════════════════════════

def compute_dnsmos(wav_path):
    if not os.path.exists(DNSMOS_ONNX):
        return None
    try:
        import onnxruntime as ort
        import librosa
        sess = ort.InferenceSession(DNSMOS_ONNX, providers=["CPUExecutionProvider"])
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        EXPECTED = 144160
        audio = audio.astype(np.float32)
        if len(audio) < EXPECTED:
            audio = np.pad(audio, (0, EXPECTED - len(audio)))
        scores = []
        for start in range(0, max(1, len(audio) - EXPECTED + 1), EXPECTED):
            seg = audio[start:start + EXPECTED]
            if len(seg) < EXPECTED:
                seg = np.pad(seg, (0, EXPECTED - len(seg)))
            out = sess.run(None, {sess.get_inputs()[0].name: seg[np.newaxis, :]})[0][0]
            scores.append(out)
        avg = np.mean(scores, axis=0)
        return {"SIG": round(float(avg[0]), 3),
                "BAK": round(float(avg[1]), 3),
                "OVRL": round(float(avg[2]), 3)}
    except Exception as e:
        print(f"  [DNSMOS ERR] {e}")
        return None


def snr_estimate(wav_path):
    try:
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        frame = int(sr * 0.02)
        energies = [np.sum(audio[i:i+frame]**2) for i in range(0, len(audio)-frame, frame)]
        if not energies:
            return None
        energies = sorted(energies)
        noise = np.mean(energies[:max(1, len(energies)//5)]) + 1e-12
        signal = np.mean(energies[len(energies)//2:]) + 1e-12
        return round(float(10 * np.log10(signal / noise)), 1)
    except:
        return None


# ═══════════════════════════════════════════════════════════════════
# BƯỚC 1: Music Source Separation với audio-separator
# ═══════════════════════════════════════════════════════════════════

def run_separation(wav_files, model_short_name, model_filename):
    print(f"\n{'='*60}")
    print(f"BƯỚC 1 - TÁCH VOCAL: {model_short_name}")
    print(f"  Model file: {model_filename}")
    print("="*60)

    out_dir = os.path.join(SEP_OUT_DIR, model_short_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    results = {}
    try:
        from audio_separator.separator import Separator
        sep = Separator(
            output_dir=out_dir,
            model_file_dir=MODEL_DIR,
            output_format="WAV",
            normalization_threshold=0.9,
        )
        sep.load_model(model_filename)

        for wav_path in wav_files:
            name = os.path.splitext(os.path.basename(wav_path))[0]

            # audio-separator thay __ bằng _ trong tên file output
            name_sanitized = name.replace("__", "_")
            model_stem = os.path.splitext(model_filename)[0]

            # Tìm file vocals đã có sẵn (nếu đã chạy trước đó)
            # Ưu tiên tìm stem có "(Vocals)" hoặc "(vocals)" trước tên model
            existing_vocal = None
            candidates = []
            for f in os.listdir(out_dir):
                fl = f.lower()
                if name_sanitized.lower() in fl:
                    # Chỉ lấy stem vocals, không lấy (other) hay (instrumental)
                    if "(vocals)" in fl:
                        candidates.insert(0, os.path.join(out_dir, f))  # ưu tiên cao nhất
                    elif "(instrumental)" not in fl and "(other)" not in fl and "vocal" in fl:
                        candidates.append(os.path.join(out_dir, f))
            if candidates:
                existing_vocal = candidates[0]

            if existing_vocal:
                print(f"  [{name}] SKIP (đã có sẵn): {os.path.basename(existing_vocal)}")
                audio, sr = sf.read(wav_path)
                duration = audio.shape[0] / sr
                results[name] = {
                    "model": model_short_name,
                    "vocal_path": existing_vocal,
                    "all_stems": [existing_vocal],
                    "duration_s": round(duration, 2),
                    "process_time_s": None,
                    "RTF": None,
                }
                continue

            print(f"  [{name}] đang tách vocal...")
            audio, sr = sf.read(wav_path)
            duration = audio.shape[0] / sr

            t0 = time.perf_counter()
            out_files = sep.separate(wav_path)
            elapsed = time.perf_counter() - t0
            rtf = elapsed / max(duration, 0.001)

            # audio-separator có thể trả về tên file tương đối → cần build full path
            out_files_full = []
            for f in out_files:
                if not os.path.isabs(f):
                    # Thử tìm trong out_dir trước
                    candidate = os.path.join(out_dir, f)
                    if os.path.exists(candidate):
                        f = candidate
                    else:
                        # Thử tìm trực tiếp từ working dir
                        if not os.path.exists(f):
                            f = candidate  # fallback
                out_files_full.append(f)
            out_files = out_files_full

            # Tìm file vocals trong output
            vocal_path = None
            for f in out_files:
                fname_lower = os.path.basename(f).lower()
                if "vocal" in fname_lower or "(vocals)" in fname_lower:
                    vocal_path = f
                    break
            if vocal_path is None and out_files:
                vocal_path = out_files[0]

            results[name] = {
                "model": model_short_name,
                "vocal_path": vocal_path,
                "all_stems": out_files,
                "duration_s": round(duration, 2),
                "process_time_s": round(elapsed, 3),
                "RTF": round(rtf, 4),
            }
            stem_names = [os.path.basename(f) for f in out_files]
            print(f"    ✓ RTF={rtf:.3f} | stems: {stem_names}")

    except Exception as e:
        print(f"  [ERROR] {model_short_name}: {e}")
        import traceback; traceback.print_exc()

    return results


# ═══════════════════════════════════════════════════════════════════
# BƯỚC 2: Speech Enhancement (DeepFilterNet 3) trên vocal đã tách
# ═══════════════════════════════════════════════════════════════════

def run_enhancement_on_vocals(separation_results, model_short_name):
    print(f"\n{'='*60}")
    print(f"BƯỚC 2 - SPEECH ENHANCEMENT trên vocal ({model_short_name})")
    print("="*60)

    out_dir = os.path.join(ENH_OUT_DIR, model_short_name)
    os.makedirs(out_dir, exist_ok=True)

    enhanced_results = {}
    try:
        import torch
        import types, torchaudio
        _backend_mod = types.ModuleType("torchaudio.backend")
        _common_mod  = types.ModuleType("torchaudio.backend.common")
        from collections import namedtuple
        _common_mod.AudioMetaData = namedtuple(
            "AudioMetaData", ["sample_rate","num_frames","num_channels","bits_per_sample","encoding"]
        )
        sys.modules["torchaudio.backend"]        = _backend_mod
        sys.modules["torchaudio.backend.common"] = _common_mod

        from df import enhance, init_df
        model, df_state, _ = init_df()
        model_sr = df_state.sr()

        for name, info in separation_results.items():
            vocal_path = info.get("vocal_path")
            if not vocal_path or not os.path.exists(vocal_path):
                print(f"  [{name}] SKIP – không tìm thấy vocals path")
                continue

            out_name = f"{name}__{model_short_name}__dfn3.wav"
            out_path = os.path.join(out_dir, out_name)

            print(f"  [{name}] enhance...")
            audio, sr = sf.read(vocal_path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            audio_tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            if sr != model_sr:
                import torchaudio.functional as F
                audio_tensor = F.resample(audio_tensor, sr, model_sr)

            t0 = time.perf_counter()
            enhanced = enhance(model, df_state, audio_tensor)
            elapsed = time.perf_counter() - t0
            duration = audio_tensor.shape[-1] / sr
            rtf = elapsed / max(duration, 0.001)

            sf.write(out_path, enhanced.squeeze().numpy(), model_sr)
            enhanced_results[name] = {
                "enhanced_path": out_path,
                "duration_s": round(duration, 2),
                "enhance_time_s": round(elapsed, 3),
                "RTF_enhance": round(rtf, 4),
            }
            print(f"    ✓ RTF={rtf:.3f} | saved: {os.path.basename(out_path)}")

    except Exception as e:
        print(f"  [ERROR] DeepFilterNet: {e}")
        import traceback; traceback.print_exc()

    return enhanced_results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█"*65)
    print("  MUSIC SOURCE SEPARATION + SPEECH ENHANCEMENT PIPELINE")
    print("  Dataset : YouTube audio (khanhvy_vlog, khanhvy_talkshow, beyondlimits)")
    print("  MSS     : BS-Roformer-Viperx-1297  vs  MelBand-Roformer-Vocal")
    print("  Enhance : DeepFilterNet 3")
    print("  Eval    : DNSMOS P.835 (SIG, BAK, OVRL)")
    print("█"*65)

    # ── Bước 0: Chuẩn bị WAV ──────────────────────────────────────
    wav_files = extract_and_convert()
    if not wav_files:
        print("[ERROR] Không tìm thấy file WAV nào!")
        return

    print(f"\n[INFO] Tổng cộng {len(wav_files)} file WAV:")
    for wf in wav_files:
        kb = os.path.getsize(wf) / 1024
        dur_info = ""
        try:
            a, sr = sf.read(wf)
            dur_info = f" | {a.shape[0]/sr:.1f}s"
        except:
            pass
        print(f"  - {os.path.basename(wf)} ({kb:.0f} KB{dur_info})")

    # ── DNSMOS baseline (raw audio) ────────────────────────────────
    print("\n[BASELINE] DNSMOS & SNR trên audio GỐC (raw webm→wav):")
    raw_metrics = {}
    for wf in wav_files:
        name = os.path.splitext(os.path.basename(wf))[0]
        snr  = snr_estimate(wf)
        dnsmos = compute_dnsmos(wf)
        raw_metrics[name] = {"snr": snr, "dnsmos": dnsmos}
        if dnsmos:
            print(f"  {name[:40]:<40} SNR={snr:>6} dB | OVRL={dnsmos['OVRL']} SIG={dnsmos['SIG']} BAK={dnsmos['BAK']}")

    # ── Chạy 2 separation models ───────────────────────────────────
    all_results = {}
    for model_short, model_file in SEPARATION_MODELS.items():
        sep_results  = run_separation(wav_files, model_short, model_file)
        enh_results  = run_enhancement_on_vocals(sep_results, model_short)

        # DNSMOS sau tách vocal (trước enhance)
        print(f"\n[DNSMOS] Sau tách vocal ({model_short}) – trước enhance:")
        for name, info in sep_results.items():
            vp = info.get("vocal_path")
            scores = compute_dnsmos(vp) if vp and os.path.exists(vp) else None
            sep_results[name]["dnsmos_vocal"] = scores
            snr_v = snr_estimate(vp) if vp and os.path.exists(vp) else None
            sep_results[name]["snr_vocal"] = snr_v
            if scores:
                print(f"  {name[:40]:<40} SNR={snr_v:>6} | OVRL={scores['OVRL']} SIG={scores['SIG']} BAK={scores['BAK']}")

        # DNSMOS sau enhance
        print(f"\n[DNSMOS] Sau enhance ({model_short} → DFN3):")
        for name, info in enh_results.items():
            ep = info.get("enhanced_path")
            scores = compute_dnsmos(ep) if ep and os.path.exists(ep) else None
            enh_results[name]["dnsmos_enhanced"] = scores
            if scores:
                print(f"  {name[:40]:<40} OVRL={scores['OVRL']} SIG={scores['SIG']} BAK={scores['BAK']}")

        all_results[model_short] = {
            "separation": sep_results,
            "enhancement": enh_results,
        }

    # ── Bảng kết quả tổng hợp ─────────────────────────────────────
    print("\n" + "="*95)
    print("BẢNG KẾT QUẢ TỔNG HỢP")
    print("="*95)

    col = f"{'File':<28} {'SNR_raw':>8} {'OVRL_raw':>9}"
    for model_short in SEPARATION_MODELS:
        col += f" {'OVRL_sep_'+model_short[:8]:>14} {'OVRL_enh_'+model_short[:8]:>14}"
    print(col)
    print("-"*95)

    final_rows = []
    for wf in wav_files:
        name = os.path.splitext(os.path.basename(wf))[0]
        rm   = raw_metrics.get(name, {})
        snr_r = rm.get("snr")
        ov_raw = rm.get("dnsmos", {}).get("OVRL") if rm.get("dnsmos") else None

        row = {"file": name, "snr_raw": snr_r, "dnsmos_raw": rm.get("dnsmos")}
        line = f"{name[:28]:<28} {str(snr_r):>8} {str(ov_raw):>9}"

        for model_short in SEPARATION_MODELS:
            sep_r = all_results.get(model_short, {}).get("separation", {}).get(name, {})
            enh_r = all_results.get(model_short, {}).get("enhancement", {}).get(name, {})
            ov_sep = sep_r.get("dnsmos_vocal", {}).get("OVRL") if sep_r.get("dnsmos_vocal") else None
            ov_enh = enh_r.get("dnsmos_enhanced", {}).get("OVRL") if enh_r.get("dnsmos_enhanced") else None
            row[f"dnsmos_sep_{model_short}"]  = sep_r.get("dnsmos_vocal")
            row[f"dnsmos_enh_{model_short}"]  = enh_r.get("dnsmos_enhanced")
            row[f"rtf_sep_{model_short}"]     = sep_r.get("RTF")
            row[f"rtf_enh_{model_short}"]     = enh_r.get("RTF_enhance")
            line += f" {str(ov_sep):>14} {str(ov_enh):>14}"

        print(line)
        final_rows.append(row)

    print("="*95)
    print("\nGhi chú:")
    print("  OVRL_raw   = DNSMOS Overall trên audio gốc (chứa nhạc nền)")
    print("  OVRL_sep   = DNSMOS Overall sau tách vocal (MSS)")
    print("  OVRL_enh   = DNSMOS Overall sau tách vocal + DeepFilterNet3")
    print("  RTF < 1.0  = xử lý nhanh hơn real-time")
    print("  DNSMOS thang 1–5 (càng cao càng tốt)")

    # ── RTF tổng hợp ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("TỐC ĐỘ XỬ LÝ (RTF – Real-Time Factor, thấp = tốt)")
    print("="*60)
    for model_short, model_file in SEPARATION_MODELS.items():
        sep_results = all_results.get(model_short, {}).get("separation", {})
        rtfs = [v.get("RTF") for v in sep_results.values() if v.get("RTF") is not None]
        if rtfs:
            print(f"  [{model_short}] RTF trung bình: {np.mean(rtfs):.3f} (min={min(rtfs):.3f}, max={max(rtfs):.3f})")

    # ── Lưu kết quả ────────────────────────────────────────────────
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(final_rows, f, ensure_ascii=False, indent=2)
    print(f"\n[XONG] Kết quả JSON lưu tại: {RESULTS_JSON}")
    print(f"[XONG] Audio tách vocal: {SEP_OUT_DIR}")
    print(f"[XONG] Audio enhanced  : {ENH_OUT_DIR}")


if __name__ == "__main__":
    main()

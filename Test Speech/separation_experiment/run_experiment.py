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
RESULTS_JSON   = os.path.join(SCRIPT_DIR, "results.json")
LATENCY_JSON   = os.path.join(SCRIPT_DIR, "latency_results.json")
TIMINGS_CACHE  = os.path.join(SCRIPT_DIR, "timings_cache.json")
DNSMOS_ONNX    = os.path.join(SCRIPT_DIR, "sig_bak_ovr.onnx")
MODEL_DIR      = os.path.join(SCRIPT_DIR, "models")

# RTF trung bình từ lần chạy đầu (CPU) — dùng ước tính khi SKIP separation chưa có cache
LATENCY_DEFAULT_RTF = {
    "bs_roformer_1297": 9.335,
    "melband_roformer_vocal": 5.673,
}
DFN3_DEFAULT_RTF = 0.060

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
# Độ trễ xử lý local (latency)
# ═══════════════════════════════════════════════════════════════════

def load_timings_cache():
    if os.path.exists(TIMINGS_CACHE):
        with open(TIMINGS_CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_timings_cache(cache):
    with open(TIMINGS_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def make_latency_record(elapsed_s, duration_s=None, estimated=False):
    """Ghi nhận độ trễ xử lý local (không phải HTTP API)."""
    rec = {
        "latency_s": round(elapsed_s, 3),
        "latency_ms": round(elapsed_s * 1000, 1),
    }
    if duration_s and duration_s > 0:
        rec["duration_s"] = round(duration_s, 2)
        rec["rtf"] = round(elapsed_s / duration_s, 4)
    if estimated:
        rec["estimated"] = True
    return rec


def timed_dnsmos(wav_path):
    t0 = time.perf_counter()
    scores = compute_dnsmos(wav_path)
    elapsed = time.perf_counter() - t0
    latency = make_latency_record(elapsed)
    return scores, latency


# ═══════════════════════════════════════════════════════════════════
# BƯỚC 0: Giải nén & convert webm → WAV
# ═══════════════════════════════════════════════════════════════════

def extract_and_convert(n_per_cat=N_SAMPLES_PER_CATEGORY):
    """Giải nén audio.zip, lấy N file đầu mỗi category, convert sang WAV 44100Hz stereo."""
    os.makedirs(RAW_WAV_DIR, exist_ok=True)
    convert_latency = {}

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
                convert_latency[out_name] = None
                continue

            # Giải nén webm tạm thời
            webm_tmp = os.path.join(RAW_WAV_DIR, f"_tmp_{name_no_ext}.webm")
            t0 = time.perf_counter()
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
            elapsed = time.perf_counter() - t0

            if result.returncode == 0:
                size_kb = os.path.getsize(out_path) / 1024
                convert_latency[out_name] = make_latency_record(elapsed)
                print(f"  ✓ {out_name} ({size_kb:.0f} KB) | convert={elapsed:.1f}s")
                wav_files.append(out_path)
            else:
                print(f"  ✗ FAIL: {out_name}")
                print(f"    {result.stderr[-300:]}")

    return wav_files, convert_latency


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

def pick_vocal_stem(file_paths):
    """Chọn đúng stem vocal, tránh nhầm với tên model chứa 'vocals'."""
    scored = []
    for path in file_paths:
        name = os.path.basename(path).lower()
        if "(vocals)" in name:
            scored.append((0, path))
        elif "(vocal)" in name and "(instrumental)" not in name:
            scored.append((1, path))
        elif "vocal" in name and "(other)" not in name and "(instrumental)" not in name:
            scored.append((2, path))
    if scored:
        scored.sort(key=lambda x: x[0])
        return scored[0][1]
    return file_paths[0] if file_paths else None


def run_separation(wav_files, model_short_name, model_filename, timings_cache):
    print(f"\n{'='*60}")
    print(f"BƯỚC 1 - TÁCH VOCAL: {model_short_name}")
    print(f"  Model file: {model_filename}")
    print("="*60)

    out_dir = os.path.join(SEP_OUT_DIR, model_short_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    results = {}
    model_load_latency = None
    try:
        from audio_separator.separator import Separator
        sep = Separator(
            output_dir=out_dir,
            model_file_dir=MODEL_DIR,
            output_format="WAV",
            normalization_threshold=0.9,
        )
        t_load = time.perf_counter()
        sep.load_model(model_filename)
        model_load_latency = make_latency_record(time.perf_counter() - t_load)
        print(f"  [LOAD] model load: {model_load_latency['latency_s']}s")

        for wav_path in wav_files:
            name = os.path.splitext(os.path.basename(wav_path))[0]

            # audio-separator thay __ bằng _ trong tên file output
            name_sanitized = name.replace("__", "_")
            model_stem = os.path.splitext(model_filename)[0]

            # Tìm file vocals đã có sẵn (nếu đã chạy trước đó)
            existing_stems = [
                os.path.join(out_dir, f)
                for f in os.listdir(out_dir)
                if name_sanitized.lower() in f.lower()
            ]
            existing_vocal = pick_vocal_stem(existing_stems)

            if existing_vocal:
                print(f"  [{name}] SKIP (đã có sẵn): {os.path.basename(existing_vocal)}")
                audio, sr = sf.read(wav_path)
                duration = audio.shape[0] / sr
                cache_key = f"{model_short_name}/{name}"
                if cache_key in timings_cache:
                    lat = timings_cache[cache_key]
                else:
                    rtf_est = LATENCY_DEFAULT_RTF.get(model_short_name, 8.0)
                    elapsed_est = rtf_est * duration
                    lat = make_latency_record(elapsed_est, duration, estimated=True)
                    timings_cache[cache_key] = lat
                results[name] = {
                    "model": model_short_name,
                    "vocal_path": existing_vocal,
                    "all_stems": [existing_vocal],
                    "duration_s": round(duration, 2),
                    "process_time_s": lat["latency_s"],
                    "RTF": lat.get("rtf"),
                    "latency_mss": lat,
                }
                est_tag = " (ước tính)" if lat.get("estimated") else ""
                print(f"    ⏱ MSS latency={lat['latency_ms']:.0f}ms RTF={lat.get('rtf')}{est_tag}")
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

            vocal_path = pick_vocal_stem(out_files)

            lat = make_latency_record(elapsed, duration)
            timings_cache[f"{model_short_name}/{name}"] = lat
            results[name] = {
                "model": model_short_name,
                "vocal_path": vocal_path,
                "all_stems": out_files,
                "duration_s": round(duration, 2),
                "process_time_s": lat["latency_s"],
                "RTF": lat.get("rtf"),
                "latency_mss": lat,
            }
            stem_names = [os.path.basename(f) for f in out_files]
            print(f"    ✓ latency={lat['latency_ms']:.0f}ms RTF={rtf:.3f} | stems: {stem_names}")

    except Exception as e:
        print(f"  [ERROR] {model_short_name}: {e}")
        import traceback; traceback.print_exc()

    return results, model_load_latency


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
            lat = make_latency_record(elapsed, duration)
            enhanced_results[name] = {
                "enhanced_path": out_path,
                "duration_s": round(duration, 2),
                "enhance_time_s": lat["latency_s"],
                "RTF_enhance": lat.get("rtf"),
                "latency_enhance": lat,
            }
            print(f"    ✓ latency={lat['latency_ms']:.0f}ms RTF={rtf:.3f} | saved: {os.path.basename(out_path)}")

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

    timings_cache = load_timings_cache()

    # ── Bước 0: Chuẩn bị WAV ──────────────────────────────────────
    wav_files, convert_latency = extract_and_convert()
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
        dnsmos, lat_dns = timed_dnsmos(wf)
        raw_metrics[name] = {"snr": snr, "dnsmos": dnsmos, "latency_dnsmos_raw": lat_dns}
        if dnsmos:
            print(f"  {name[:40]:<40} SNR={snr:>6} dB | OVRL={dnsmos['OVRL']} | dnsmos={lat_dns['latency_ms']:.0f}ms")

    # ── Chạy 2 separation models ───────────────────────────────────
    all_results = {}
    model_load_times = {}
    for model_short, model_file in SEPARATION_MODELS.items():
        sep_results, load_lat = run_separation(wav_files, model_short, model_file, timings_cache)
        model_load_times[model_short] = load_lat
        enh_results  = run_enhancement_on_vocals(sep_results, model_short)

        # DNSMOS sau tách vocal (trước enhance)
        print(f"\n[DNSMOS] Sau tách vocal ({model_short}) – trước enhance:")
        for name, info in sep_results.items():
            vp = info.get("vocal_path")
            if vp and os.path.exists(vp):
                scores, lat_dns = timed_dnsmos(vp)
                sep_results[name]["dnsmos_vocal"] = scores
                sep_results[name]["latency_dnsmos_sep"] = lat_dns
                snr_v = snr_estimate(vp)
                sep_results[name]["snr_vocal"] = snr_v
                if scores:
                    print(f"  {name[:40]:<40} OVRL={scores['OVRL']} | dnsmos={lat_dns['latency_ms']:.0f}ms")

        # DNSMOS sau enhance
        print(f"\n[DNSMOS] Sau enhance ({model_short} → DFN3):")
        for name, info in enh_results.items():
            ep = info.get("enhanced_path")
            if ep and os.path.exists(ep):
                scores, lat_dns = timed_dnsmos(ep)
                enh_results[name]["dnsmos_enhanced"] = scores
                enh_results[name]["latency_dnsmos_enh"] = lat_dns
                if scores:
                    print(f"  {name[:40]:<40} OVRL={scores['OVRL']} | dnsmos={lat_dns['latency_ms']:.0f}ms")

        all_results[model_short] = {
            "separation": sep_results,
            "enhancement": enh_results,
        }

    save_timings_cache(timings_cache)

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
            row[f"latency_mss_ms_{model_short}"] = (sep_r.get("latency_mss") or {}).get("latency_ms")
            row[f"latency_enh_ms_{model_short}"] = (enh_r.get("latency_enhance") or {}).get("latency_ms")
            mss_ms = row[f"latency_mss_ms_{model_short}"] or 0
            enh_ms = row[f"latency_enh_ms_{model_short}"] or 0
            row[f"latency_pipeline_ms_{model_short}"] = round(mss_ms + enh_ms, 1) if (mss_ms or enh_ms) else None
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

    # ── Độ trễ / RTF tổng hợp ─────────────────────────────────────
    print("\n" + "="*80)
    print("ĐỘ TRỄ XỬ LÝ LOCAL (latency — audio local, không qua HTTP API)")
    print("="*80)
    hdr = f"{'Pipeline':<28} {'MSS ms':>10} {'Enh ms':>10} {'Total ms':>12} {'RTF MSS':>9} {'RTF Enh':>9}"
    print(hdr)
    print("-"*80)

    latency_report = {
        "note": "Độ trễ xử lý local (wall-clock). Không phải latency HTTP API.",
        "model_load": model_load_times,
        "summary": {},
        "per_file": {},
    }

    for model_short in SEPARATION_MODELS:
        sep_results = all_results.get(model_short, {}).get("separation", {})
        enh_results = all_results.get(model_short, {}).get("enhancement", {})

        mss_ms_list = []
        enh_ms_list = []
        rtfs_sep = []
        rtfs_enh = []

        for name in sep_results:
            sep_r = sep_results.get(name, {})
            enh_r = enh_results.get(name, {})
            mss_lat = sep_r.get("latency_mss") or {}
            enh_lat = enh_r.get("latency_enhance") or {}
            mss_ms = mss_lat.get("latency_ms")
            enh_ms = enh_lat.get("latency_ms")
            if mss_ms:
                mss_ms_list.append(mss_ms)
            if enh_ms:
                enh_ms_list.append(enh_ms)
            if sep_r.get("RTF") is not None:
                rtfs_sep.append(sep_r["RTF"])
            if enh_r.get("RTF_enhance") is not None:
                rtfs_enh.append(enh_r["RTF_enhance"])

            if name not in latency_report["per_file"]:
                latency_report["per_file"][name] = {}
            latency_report["per_file"][name][model_short] = {
                "mss": mss_lat,
                "enhance": enh_lat,
                "dnsmos_sep": sep_r.get("latency_dnsmos_sep"),
                "dnsmos_enh": enh_r.get("latency_dnsmos_enh"),
                "pipeline_total_ms": round((mss_ms or 0) + (enh_ms or 0), 1) if (mss_ms or enh_ms) else None,
            }

        if mss_ms_list:
            latency_report["summary"][model_short] = {
                "mss_latency_ms_avg": round(np.mean(mss_ms_list), 1),
                "mss_latency_ms_min": round(min(mss_ms_list), 1),
                "mss_latency_ms_max": round(max(mss_ms_list), 1),
                "enhance_latency_ms_avg": round(np.mean(enh_ms_list), 1) if enh_ms_list else None,
                "pipeline_latency_ms_avg": round(np.mean(mss_ms_list) + np.mean(enh_ms_list), 1) if enh_ms_list else round(np.mean(mss_ms_list), 1),
                "rtf_mss_avg": round(np.mean(rtfs_sep), 3) if rtfs_sep else None,
                "rtf_enhance_avg": round(np.mean(rtfs_enh), 4) if rtfs_enh else None,
            }
            s = latency_report["summary"][model_short]
            print(
                f"{model_short:<28} "
                f"{s['mss_latency_ms_avg']:>10.0f} "
                f"{s.get('enhance_latency_ms_avg') or 0:>10.0f} "
                f"{s['pipeline_latency_ms_avg']:>12.0f} "
                f"{s.get('rtf_mss_avg') or 0:>9.3f} "
                f"{s.get('rtf_enhance_avg') or 0:>9.4f}"
            )

    print("="*80)
    print("  latency_ms = thời gian xử lý thực tế (giây × 1000)")
    print("  Total ms   = MSS + DeepFilterNet3 (chưa gồm DNSMOS)")
    print("  RTF < 1    = nhanh hơn real-time")

    # ── Lưu kết quả ────────────────────────────────────────────────
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(final_rows, f, ensure_ascii=False, indent=2)
    with open(LATENCY_JSON, "w", encoding="utf-8") as f:
        json.dump(latency_report, f, ensure_ascii=False, indent=2)
    print(f"\n[XONG] Kết quả JSON lưu tại: {RESULTS_JSON}")
    print(f"[XONG] Độ trễ JSON lưu tại: {LATENCY_JSON}")
    print(f"[XONG] Timings cache   : {TIMINGS_CACHE}")
    print(f"[XONG] Audio tách vocal: {SEP_OUT_DIR}")
    print(f"[XONG] Audio enhanced  : {ENH_OUT_DIR}")


if __name__ == "__main__":
    main()

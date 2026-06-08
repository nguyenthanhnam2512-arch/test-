# -*- coding: utf-8 -*-
"""
Perceptual / Voice-Distortion Evaluation
==========================================
Đánh giá chất lượng giọng nói sau khi qua MSS + Speech Enhancement.
Vấn đề chính cần phát hiện: "méo giọng" (voice distortion / artifacts).

Metrics tính được:
  1. F0 RMSE          — độ lệch tần số cơ bản (pitch); cao = méo giọng
  2. MCD              — Mel Cepstral Distortion; cao = thay đổi âm sắc lớn
  3. EPR              — Energy Preservation Ratio; thấp = mất năng lượng giọng
  4. SFR              — Spectral Flatness Ratio; cao = giọng bị phẳng/robot
  5. VUV shift        — Thay đổi tỉ lệ voiced/unvoiced
  6. VDS (tổng hợp)   — Voice Distortion Score (0–10, thấp là tốt)

So sánh: raw vs (BS-Roformer+DFN3) vs (MelBand-Roformer+DFN3)
Đây là complement của DNSMOS: DNSMOS đo background noise suppression,
trong khi VDS đo mức độ méo giọng của chính người nói.

Ref paper so sánh: SIDON (arXiv:2509.17052v3) — Nakata et al., 2026
"""

import sys, os, io, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import librosa

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RAW_DIR     = os.path.join(SCRIPT_DIR, "audio_raw_wav")
SEP_DIR     = os.path.join(SCRIPT_DIR, "audio_separated")
ENH_DIR     = os.path.join(SCRIPT_DIR, "audio_enhanced")
OUT_JSON    = os.path.join(SCRIPT_DIR, "perceptual_results.json")

MODELS = {
    "bs_roformer_1297":      "BS-Roformer-Viperx-1297 + DFN3",
    "melband_roformer_vocal": "Mel-Band-Roformer + DFN3",
}

# Pattern tên file enhanced (từ run_experiment.py)
ENH_SUFFIX = {
    "bs_roformer_1297":       "__bs_roformer_1297__dfn3.wav",
    "melband_roformer_vocal":  "__melband_roformer_vocal__dfn3.wav",
}

# Pattern tên file separated vocal (từ audio-separator output)
# Raw: {cat}__{id}.wav  →  Separated: {cat}_{id}_(Vocals)_model_bs_roformer_ep_317_sdr_12.wav
SEP_VOCAL_SUFFIX = {
    "bs_roformer_1297":      "_(Vocals)_model_bs_roformer_ep_317_sdr_12.wav",
    "melband_roformer_vocal": "_(vocals)_vocals_mel_band_roformer.wav",
}

SR_ANALYSIS = 16000   # downsample để tính metrics
N_MFCC      = 20      # số MFCC coefficients cho MCD
F0_FMIN     = 60.0    # Hz — tránh false detection
F0_FMAX     = 500.0   # Hz

# ══════════════════════════════════════════════════════════════════
# Helper: load & resample
# ══════════════════════════════════════════════════════════════════

def load_mono(path, sr=SR_ANALYSIS):
    audio, orig_sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if orig_sr != sr:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=orig_sr, target_sr=sr)
    return audio.astype(np.float32), sr


def trim_to_same_length(a, b):
    """Cắt 2 mảng về độ dài tối thiểu để so sánh được."""
    n = min(len(a), len(b))
    return a[:n], b[:n]


def normalize_audio(a):
    """Normalize âm lượng để so sánh spectral (peak normalization)."""
    peak = np.max(np.abs(a))
    if peak > 1e-6:
        return a / peak
    return a

# ══════════════════════════════════════════════════════════════════
# 1. F0 RMSE  — pitch deviation
# ══════════════════════════════════════════════════════════════════

def compute_f0_rmse(raw, enh, sr):
    """
    Tính RMSE của F0 (Hz) giữa raw và enhanced trên các frame voiced.
    Kết quả cao → pitch bị lệch → méo giọng.
    """
    try:
        f0_raw, voiced_flag_raw, _ = librosa.pyin(
            raw, fmin=F0_FMIN, fmax=F0_FMAX, sr=sr,
            frame_length=2048, hop_length=512
        )
        f0_enh, voiced_flag_enh, _ = librosa.pyin(
            enh, fmin=F0_FMIN, fmax=F0_FMAX, sr=sr,
            frame_length=2048, hop_length=512
        )
        n = min(len(f0_raw), len(f0_enh))
        f0_raw, f0_enh = f0_raw[:n], f0_enh[:n]
        voiced_raw = ~np.isnan(f0_raw)
        voiced_enh = ~np.isnan(f0_enh)
        voiced_both = voiced_raw & voiced_enh

        if voiced_both.sum() < 10:
            return None, None, None

        # RMSE trên voiced frames
        rmse = float(np.sqrt(np.mean((f0_raw[voiced_both] - f0_enh[voiced_both])**2)))

        # VUV shift: thay đổi tỉ lệ voiced
        vuv_raw = voiced_raw.mean()
        vuv_enh = voiced_enh.mean()
        vuv_shift = float(abs(vuv_raw - vuv_enh))

        return round(rmse, 2), round(float(vuv_raw), 3), round(vuv_shift, 3)
    except Exception as e:
        print(f"    [F0 ERR] {e}")
        return None, None, None

# ══════════════════════════════════════════════════════════════════
# 2. MCD — Mel Cepstral Distortion
# ══════════════════════════════════════════════════════════════════

def compute_mcd(raw, enh, sr):
    """
    Mel Cepstral Distortion (dB). Thông thường MCD < 4 dB là tốt.
    Cao → âm sắc bị thay đổi nhiều → méo giọng.
    """
    try:
        mfcc_raw = librosa.feature.mfcc(y=raw, sr=sr, n_mfcc=N_MFCC, n_fft=512, hop_length=256)
        mfcc_enh = librosa.feature.mfcc(y=enh, sr=sr, n_mfcc=N_MFCC, n_fft=512, hop_length=256)

        n = min(mfcc_raw.shape[1], mfcc_enh.shape[1])
        mfcc_raw, mfcc_enh = mfcc_raw[:, :n], mfcc_enh[:, :n]

        # MCD = (10/ln10) * sqrt(2 * sum((c_raw - c_enh)^2)) per frame, then mean
        diff = mfcc_raw[1:] - mfcc_enh[1:]   # skip c0 (energy)
        mcd_per_frame = (10.0 / np.log(10.0)) * np.sqrt(2.0 * np.sum(diff**2, axis=0))
        mcd = float(np.mean(mcd_per_frame))
        return round(mcd, 3)
    except Exception as e:
        print(f"    [MCD ERR] {e}")
        return None

# ══════════════════════════════════════════════════════════════════
# 3. Energy Preservation Ratio (EPR)
# ══════════════════════════════════════════════════════════════════

def compute_epr(raw, enh):
    """
    Tỉ lệ năng lượng được giữ lại sau xử lý.
    EPR ~1.0 là lý tưởng. EPR < 0.5 = mất giọng nhiều.
    """
    try:
        e_raw = float(np.mean(raw**2)) + 1e-12
        e_enh = float(np.mean(enh**2)) + 1e-12
        return round(e_enh / e_raw, 4)
    except:
        return None

# ══════════════════════════════════════════════════════════════════
# 4. Spectral Flatness Ratio (SFR)
# ══════════════════════════════════════════════════════════════════

def compute_sfr(raw, enh, sr):
    """
    So sánh spectral flatness (độ phẳng phổ) giữa raw và enhanced.
    SFR > 1.5 → giọng bị "phẳng hơn" (mất harmonics → méo giọng).
    """
    try:
        sf_raw = librosa.feature.spectral_flatness(y=raw, hop_length=256)
        sf_enh = librosa.feature.spectral_flatness(y=enh, hop_length=256)
        mean_raw = float(np.mean(sf_raw)) + 1e-12
        mean_enh = float(np.mean(sf_enh)) + 1e-12
        sfr = mean_enh / mean_raw
        return round(sfr, 4), round(float(np.log10(mean_raw * 1000 + 1)), 4)
    except:
        return None, None

# ══════════════════════════════════════════════════════════════════
# 5. Voice Distortion Score (VDS) — composite
# ══════════════════════════════════════════════════════════════════

def compute_vds(f0_rmse, mcd, epr, sfr):
    """
    VDS ∈ [0, 10] — điểm tổng hợp méo giọng (thấp = ít méo = tốt).
    Trọng số: F0 (30%), MCD (30%), EPR (20%), SFR (20%).
    """
    score = 0.0
    count = 0

    # F0 RMSE: chuẩn hoá — baseline ~15 Hz bình thường; >50 Hz là bad
    if f0_rmse is not None:
        f0_score = min(f0_rmse / 50.0, 1.0) * 10.0
        score += 0.30 * f0_score
        count += 0.30

    # MCD (librosa scale, sep vs enh, normalized):
    # Observed range: <20 dB = ít thay đổi, >70 dB = thay đổi nhiều
    if mcd is not None:
        mcd_score = min(max(mcd - 20.0, 0.0) / 50.0, 1.0) * 10.0
        score += 0.30 * mcd_score
        count += 0.30

    # EPR: 1.0 = hoàn hảo, <0.3 = mất giọng
    if epr is not None:
        epr_score = max(1.0 - epr, 0.0) / 0.7 * 10.0
        epr_score = min(epr_score, 10.0)
        score += 0.20 * epr_score
        count += 0.20

    # SFR: 1.0 = không đổi; >2.0 = rất flat
    if sfr is not None:
        sfr_score = min(max(sfr - 1.0, 0.0) / 2.0, 1.0) * 10.0
        score += 0.20 * sfr_score
        count += 0.20

    if count == 0:
        return None
    return round(score / count, 2)

# ══════════════════════════════════════════════════════════════════
# 6. Nhận xét tự động về méo giọng
# ══════════════════════════════════════════════════════════════════

def auto_verdict(f0_rmse, mcd, epr, sfr_val, vds):
    """Sinh nhận xét tự động về chất lượng giọng."""
    issues = []
    good_points = []

    if f0_rmse is not None:
        if f0_rmse > 30:
            issues.append(f"Pitch lệch cao ({f0_rmse:.1f} Hz RMSE) → khả năng méo giọng")
        elif f0_rmse > 15:
            issues.append(f"Pitch hơi lệch ({f0_rmse:.1f} Hz RMSE)")
        else:
            good_points.append(f"Pitch ổn định ({f0_rmse:.1f} Hz RMSE)")

    if mcd is not None:
        if mcd > 70:
            issues.append(f"MCD cao ({mcd:.1f}) → DFN3 thay đổi phổ nhiều (heavy background)")
        elif mcd > 30:
            issues.append(f"MCD trung bình ({mcd:.1f}) → DFN3 hơi thay đổi âm sắc")
        else:
            good_points.append(f"MCD thấp ({mcd:.1f}) → DFN3 giữ âm sắc tốt")

    if epr is not None:
        if epr < 0.4:
            issues.append(f"Mất nhiều năng lượng giọng (EPR={epr:.2f})")
        elif epr < 0.7:
            issues.append(f"Mất một ít năng lượng (EPR={epr:.2f})")
        else:
            good_points.append(f"Năng lượng giọng được giữ tốt (EPR={epr:.2f})")

    if sfr_val is not None:
        if sfr_val > 1.8:
            issues.append(f"Phổ bị phẳng nhiều (SFR={sfr_val:.2f}) → giọng robot")
        elif sfr_val > 1.3:
            issues.append(f"Phổ hơi phẳng hơn (SFR={sfr_val:.2f})")

    if vds is not None:
        if vds <= 2.5:
            verdict = "✅ Rất ít méo giọng — phù hợp production"
        elif vds <= 4.5:
            verdict = "🟡 Méo giọng nhẹ — chấp nhận được"
        elif vds <= 6.5:
            verdict = "🟠 Méo giọng trung bình — cần kiểm tra thủ công"
        else:
            verdict = "🔴 Méo giọng nặng — không nên dùng cho TTS training"
    else:
        verdict = "N/A"

    return {
        "verdict": verdict,
        "issues": issues,
        "good_points": good_points,
    }

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█"*65)
    print("  PERCEPTUAL / VOICE DISTORTION EVALUATION")
    print("  Metrics: F0-RMSE | MCD | EPR | SFR | VDS")
    print("█"*65)

    raw_files = sorted([
        f for f in os.listdir(RAW_DIR) if f.endswith(".wav")
    ])
    print(f"\n[INFO] {len(raw_files)} file raw WAV tìm thấy.\n")

    all_results = []

    for raw_fname in raw_files:
        base = os.path.splitext(raw_fname)[0]
        raw_path = os.path.join(RAW_DIR, raw_fname)
        print(f"{'─'*60}")
        print(f"[FILE] {base}")

        raw_audio, sr = load_mono(raw_path)
        rec = {"file": base, "models": {}}

        # base_single = base với double-underscore → single (cho tên file separated)
        base_single = base.replace("__", "_")

        for model_key, model_label in MODELS.items():
            enh_fname = f"{base}{ENH_SUFFIX[model_key]}"
            enh_path  = os.path.join(ENH_DIR, model_key, enh_fname)

            # Tìm file separated vocal
            sep_fname = f"{base_single}{SEP_VOCAL_SUFFIX[model_key]}"
            sep_path  = os.path.join(SEP_DIR, model_key, sep_fname)

            if not os.path.exists(enh_path):
                print(f"  [{model_key}] SKIP — file không tồn tại: {enh_fname}")
                continue

            print(f"\n  Model: {model_label}")
            enh_audio, _ = load_mono(enh_path)

            # ── Chọn reference cho so sánh ──────────────────────
            # Ưu tiên: so sánh SEP vs ENH (chỉ đo distortion của DFN3)
            # Fallback: so sánh RAW vs ENH (khi không có sep file)
            if os.path.exists(sep_path):
                sep_audio, _ = load_mono(sep_path)
                ref_audio, cmp_audio = trim_to_same_length(sep_audio, enh_audio)
                compare_mode = "sep_vs_enh"
            else:
                ref_audio, cmp_audio = trim_to_same_length(raw_audio, enh_audio)
                compare_mode = "raw_vs_enh"
                print(f"    [INFO] Không có sep file, dùng raw vs enh")

            # Normalize trước khi so sánh spectral metrics
            ref_n = normalize_audio(ref_audio)
            cmp_n = normalize_audio(cmp_audio)

            # ── Metrics ──────────────────────────────────────────
            # F0: compare normalized (pitch không phụ thuộc amplitude)
            f0_rmse, vuv_raw, vuv_shift = compute_f0_rmse(ref_n, cmp_n, sr)
            # MCD: compare normalized (spectral envelope, loại trừ energy diff)
            mcd   = compute_mcd(ref_n, cmp_n, sr)
            # EPR: dùng raw vs enh để đo energy preservation thực tế
            raw_trim, enh_trim = trim_to_same_length(raw_audio, enh_audio)
            epr   = compute_epr(raw_trim, enh_trim)
            # SFR: compare normalized
            sfr, sfr_raw_log = compute_sfr(ref_n, cmp_n, sr)
            vds   = compute_vds(f0_rmse, mcd, epr, sfr)
            notes = auto_verdict(f0_rmse, mcd, epr, sfr, vds)

            row = {
                "model_key":     model_key,
                "model_label":   model_label,
                "compare_mode":  compare_mode,
                "f0_rmse_hz":    f0_rmse,
                "vuv_ratio_ref": vuv_raw,
                "vuv_shift":     vuv_shift,
                "mcd_dB":        mcd,
                "epr":           epr,
                "sfr":           sfr,
                "vds":           vds,
                "verdict":       notes["verdict"],
                "issues":        notes["issues"],
                "good_points":   notes["good_points"],
            }
            rec["models"][model_key] = row

            # Print
            print(f"    F0-RMSE : {f0_rmse} Hz  (VUV raw={vuv_raw}, shift={vuv_shift})")
            print(f"    MCD     : {mcd} dB")
            print(f"    EPR     : {epr}  (Energy Preservation)")
            print(f"    SFR     : {sfr}  (Spectral Flatness Ratio)")
            print(f"    VDS     : {vds}/10  → {notes['verdict']}")
            if notes["issues"]:
                for iss in notes["issues"]:
                    print(f"      ⚠ {iss}")
            if notes["good_points"]:
                for gp in notes["good_points"]:
                    print(f"      ✓ {gp}")

        all_results.append(rec)

    # ── Lưu JSON ──────────────────────────────────────────────────
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n{'█'*65}")
    print(f"[XONG] Kết quả lưu tại: {OUT_JSON}")

    # ── Bảng tóm tắt ──────────────────────────────────────────────
    print("\n" + "="*85)
    print(f"{'File':<40} {'Model':<28} {'MCD':>6} {'F0-RMSE':>8} {'EPR':>5} {'VDS':>5}")
    print("-"*85)
    for rec in all_results:
        f = rec["file"]
        for mk, mv in rec["models"].items():
            label = "BS-RoF+DFN3" if mk == "bs_roformer_1297" else "MelB+DFN3"
            mcd_s  = f"{mv['mcd_dB']:.1f}" if mv['mcd_dB'] else "N/A"
            f0_s   = f"{mv['f0_rmse_hz']:.1f}" if mv['f0_rmse_hz'] else "N/A"
            epr_s  = f"{mv['epr']:.2f}" if mv['epr'] else "N/A"
            vds_s  = f"{mv['vds']:.1f}" if mv['vds'] else "N/A"
            print(f"{f:<40} {label:<28} {mcd_s:>6} {f0_s:>8} {epr_s:>5} {vds_s:>5}")
    print("="*85)
    print("\nGhi chú:")
    print("  MCD (dB)   : đo sep_vocal vs enhanced (DFN3 distortion, librosa scale) — < 20 tốt | 20-70 trung bình | > 70 tệ")
    print("  F0-RMSE(Hz): pitch lệch sep vs enh — < 15 tốt | 15-30 trung bình | > 30 méo giọng")
    print("  EPR        : raw vs enh energy ratio — ~0.3-0.8 bình thường (music removed) | < 0.2 = mất giọng")
    print("  VDS (0-10) : < 2.5 tốt | 2.5-4.5 chấp nhận | > 4.5 méo giọng (cần nghe thủ công)")

    return all_results


if __name__ == "__main__":
    main()

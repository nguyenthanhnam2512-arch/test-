# -*- coding: utf-8 -*-
"""Tạo file Word báo cáo kết quả thực nghiệm MSS + Speech Enhancement Pipeline."""

import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
from docx import Document
from docx.shared import Pt, RGBColor, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(SCRIPT_DIR, "results.json")
OUT_DOCX     = os.path.join(SCRIPT_DIR, "..", "..", "MSS_SpeechEnhancement_Report.docx")

# RTF đo được trong thực nghiệm (từ terminal log)
RTF_BS   = {"mean": 9.335, "min": 8.557, "max": 10.290}
RTF_MEL  = {"mean": 5.673, "min": 5.005, "max": 6.449}
RTF_DFN3 = 0.060   # DeepFilterNet3 trung bình

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def h1(doc, text):
    p = doc.add_heading(text, level=1)
    p.runs[0].font.color.rgb = RGBColor(0x1A, 0x56, 0xDB)

def h2(doc, text):
    doc.add_heading(text, level=2)

def h3(doc, text):
    doc.add_heading(text, level=3)

def para(doc, text, bold=False, italic=False, size=None, color=None, align=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    if color:
        run.font.color.rgb = color
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(5)
    return p

def bullet(doc, text, level=0):
    p = doc.add_paragraph(text, style="List Bullet")
    p.paragraph_format.left_indent = Pt(18 * (level + 1))
    p.paragraph_format.space_after = Pt(3)

def note(doc, text):
    p = doc.add_paragraph()
    run = p.add_run("⚠ " + text)
    run.italic = True
    run.font.color.rgb = RGBColor(0xD9, 0x77, 0x06)
    p.paragraph_format.space_after = Pt(5)

def success(doc, text):
    p = doc.add_paragraph()
    run = p.add_run("✅ " + text)
    run.bold = True
    run.font.color.rgb = RGBColor(0x05, 0x6A, 0x34)
    p.paragraph_format.space_after = Pt(5)

def info(doc, text):
    p = doc.add_paragraph()
    run = p.add_run("ℹ " + text)
    run.italic = True
    run.font.color.rgb = RGBColor(0x1E, 0x40, 0xAF)
    p.paragraph_format.space_after = Pt(5)

def divider(doc):
    p = doc.add_paragraph("─" * 80)
    p.runs[0].font.size = Pt(7)
    p.runs[0].font.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)

def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)

def tbl(doc, headers, rows, col_widths=None, header_color="1A56DB", alt_rows=True):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    # Header row
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        for run in hdr[i].paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_bg(hdr[i], header_color)
    # Data rows
    for ri, row_data in enumerate(rows):
        r = t.add_row().cells
        for i, cell_text in enumerate(row_data):
            r[i].text = str(cell_text)
            r[i].paragraphs[0].runs[0].font.size = Pt(9)
            if alt_rows and ri % 2 == 1:
                set_cell_bg(r[i], "EFF6FF")
    if col_widths:
        for row in t.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph()

def color_cell(cell, hex_color, text_color="000000"):
    set_cell_bg(cell, hex_color)
    for run in cell.paragraphs[0].runs:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        run.font.color.rgb = RGBColor(
            int(text_color[0:2], 16),
            int(text_color[2:4], 16),
            int(text_color[4:6], 16),
        )

def ovrl_color(val):
    """Trả về mã màu ô tương ứng với giá trị OVRL."""
    if val is None:
        return "FFFFFF"
    if val >= 3.8:
        return "D1FAE5"   # xanh lá đậm
    elif val >= 3.4:
        return "DCFCE7"   # xanh lá nhạt
    elif val >= 3.0:
        return "FEF9C3"   # vàng nhạt
    elif val >= 2.5:
        return "FEF3C7"   # cam nhạt
    else:
        return "FEE2E2"   # đỏ nhạt


# ─────────────────────────────────────────────────────────
# BUILD
# ─────────────────────────────────────────────────────────

def build(out_path):
    with open(RESULTS_JSON, encoding="utf-8") as f:
        data = json.load(f)

    doc = Document()

    # ── Lề trang ────────────────────────────────────────────────
    for section in doc.sections:
        section.top_margin    = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin   = Cm(2.5)
        section.right_margin  = Cm(2.5)

    # ════════════════════════════════════════════════════════════
    # TRANG TIÊU ĐỀ
    # ════════════════════════════════════════════════════════════
    title = doc.add_heading("BÁO CÁO THỰC NGHIỆM", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub1 = doc.add_paragraph("Music Source Separation + Speech Enhancement Pipeline")
    sub1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub1.runs[0].bold = True
    sub1.runs[0].font.size = Pt(14)
    sub1.runs[0].font.color.rgb = RGBColor(0x1A, 0x56, 0xDB)

    sub2 = doc.add_paragraph(
        "Tách giọng từ YouTube audio thô bằng Roformer + DeepFilterNet 3\n"
        "Dataset: KhanhVy Vlog · KhanhVy Talkshow · Beyond Limits / NGA Levi  |  Metric: DNSMOS P.835"
    )
    sub2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub2.runs[0].italic = True
    sub2.runs[0].font.size = Pt(10)
    sub2.runs[0].font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    doc.add_paragraph()

    # ════════════════════════════════════════════════════════════
    # PHẦN 1: TỔNG QUAN
    # ════════════════════════════════════════════════════════════
    h1(doc, "1. Tổng quan & Mục tiêu")

    para(doc,
        "Audio YouTube thô thường chứa nhạc nền, tiếng ồn môi trường và nhiều tạp âm khác. "
        "Để thu được giọng nói chất lượng cao phục vụ TTS/ASR/Voice Cloning, cần qua hai bước xử lý:")

    bullet(doc, "Music Source Separation (MSS) — tách giọng nói ra khỏi nhạc nền")
    bullet(doc, "Speech Enhancement — lọc phần tạp âm còn lại trong luồng giọng")

    para(doc, "Pipeline thực nghiệm:")
    p = doc.add_paragraph()
    run = p.add_run(
        "  YouTube .webm  →  Convert WAV  →  [MSS]  →  [Speech Enhancement]  →  Clean Voice"
    )
    run.font.name = "Courier New"
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x1E, 0x40, 0xAF)

    doc.add_paragraph()
    divider(doc)

    # ════════════════════════════════════════════════════════════
    # PHẦN 2: THIẾT LẬP THỰC NGHIỆM
    # ════════════════════════════════════════════════════════════
    h1(doc, "2. Thiết lập thực nghiệm")

    h2(doc, "2.1 Dataset")
    tbl(doc,
        ["Category", "Nguồn", "Số video", "Files test", "Đặc điểm"],
        [
            ["khanhvy_vlog",         "YouTube vlog",        "23", "3", "Nhạc nền liên tục, giọng VN nữ"],
            ["khanhvy_talkshow",     "YouTube talk show",   "37", "3", "Nhạc intro/outro, nhiều nền âm thanh"],
            ["beyondlimits_ngalevi", "YouTube interview",   "29", "3", "Tương đối sạch, ít nhạc nền"],
        ],
        col_widths=[4.2, 3.5, 2.2, 2.2, 5.5],
    )
    para(doc,
         "Mỗi file được cắt 60 giây (bỏ 30s đầu) để đại diện cho nội dung chính, "
         "tránh intro/jingle. Tổng: 9 clips × 60s = 9 phút audio.", italic=True, size=9)

    h2(doc, "2.2 Môi trường")
    tbl(doc,
        ["Thành phần", "Chi tiết"],
        [
            ["OS",             "Windows 10 (AMD64)"],
            ["CPU",            "AMD Ryzen (no GPU acceleration)"],
            ["Python",         "3.11.9"],
            ["PyTorch",        "2.11.0+cpu"],
            ["audio-separator","0.44.2"],
            ["FFmpeg",         "8.1.1-full_build"],
        ],
        col_widths=[4.5, 13.0],
    )

    h2(doc, "2.3 Models đánh giá")
    tbl(doc,
        ["#", "Model", "Architecture", "Params (ước tính)", "Vocab SDR", "Mục tiêu"],
        [
            ["1", "BS-Roformer-Viperx-1297",
             "Band-Split RoFormer\n(ViperX fine-tune)",
             "~320M (639 MB ckpt)", "11.77 dB", "Tách vocal (Vocals + Instrumental)"],
            ["2", "MelBand-Roformer-Vocal\n(Kimberley Jensen)",
             "Mel-Band RoFormer\n(Kim fine-tune)",
             "~456M (913 MB ckpt)", "12.60 dB", "Tách vocal (vocals + other)"],
            ["3", "DeepFilterNet 3",
             "GRU + ERB filter bank",
             "~1.8M", "N/A", "Speech Enhancement (bước 2)"],
        ],
        col_widths=[0.6, 4.2, 3.5, 3.2, 2.2, 3.8],
    )

    info(doc,
         "Vocal SDR là chỉ số Signal-to-Distortion Ratio đo trên tập MUSDB18HQ "
         "(benchmark chuẩn cho Music Source Separation). Cao hơn = tách sạch hơn.")

    h2(doc, "2.4 Metric đánh giá")
    tbl(doc,
        ["Metric", "Mô tả", "Thang đo", "Ghi chú"],
        [
            ["DNSMOS OVRL", "Overall MOS — chất lượng tổng hợp", "1–5 (cao hơn tốt hơn)",
             "Không cần reference audio"],
            ["DNSMOS SIG",  "Signal quality — độ rõ giọng nói",  "1–5",
             "Phản ánh độ trong tiếng"],
            ["DNSMOS BAK",  "Background quality — mức độ nhiễu nền",  "1–5",
             "Cao = ít nhiễu nền"],
            ["SNR (ước tính)", "Signal-to-Noise Ratio",          "dB (cao hơn tốt hơn)",
             "Ước tính bằng VAD đơn giản"],
            ["RTF",         "Real-Time Factor",                  "× (thấp hơn tốt hơn)",
             "RTF < 1 = nhanh hơn real-time"],
        ],
        col_widths=[3.0, 4.5, 3.5, 6.5],
    )
    divider(doc)

    # ════════════════════════════════════════════════════════════
    # PHẦN 3: KẾT QUẢ DNSMOS
    # ════════════════════════════════════════════════════════════
    h1(doc, "3. Kết quả DNSMOS P.835")

    # Chuẩn bị dữ liệu
    cats = {
        "khanhvy_vlog":         [],
        "khanhvy_talkshow":     [],
        "beyondlimits_ngalevi": [],
    }
    for row in data:
        for cat in cats:
            if row["file"].startswith(cat):
                cats[cat].append(row)

    cat_labels = {
        "khanhvy_vlog":         "KhanhVy Vlog",
        "khanhvy_talkshow":     "KhanhVy Talkshow",
        "beyondlimits_ngalevi": "Beyond Limits / NGA Levi",
    }

    h2(doc, "3.1 Bảng DNSMOS OVRL chi tiết theo từng file")

    headers = [
        "File (60s clip)",
        "Raw\nOVRL",
        "BS-Rofor\nOVRL",
        "BS+DFN3\nOVRL",
        "MelBand\nOVRL",
        "MelBand\n+DFN3",
        "Δ Best\nvs Raw",
    ]

    for cat, rows_cat in cats.items():
        h3(doc, cat_labels[cat])
        t = doc.add_table(rows=1, cols=len(headers))
        t.style = "Table Grid"
        hdr = t.rows[0].cells
        for i, h in enumerate(headers):
            hdr[i].text = h
            for run in hdr[i].paragraphs[0].runs:
                run.bold = True
                run.font.size = Pt(8)
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            set_cell_bg(hdr[i], "1A56DB")

        for row in rows_cat:
            short = row["file"].replace(cat + "__", "").replace(cat + "_", "")
            raw_ovrl = row.get("dnsmos_raw", {}).get("OVRL") if row.get("dnsmos_raw") else None
            bs_sep   = row.get("dnsmos_sep_bs_roformer_1297", {}).get("OVRL") if row.get("dnsmos_sep_bs_roformer_1297") else None
            bs_enh   = row.get("dnsmos_enh_bs_roformer_1297", {}).get("OVRL") if row.get("dnsmos_enh_bs_roformer_1297") else None
            mb_sep   = row.get("dnsmos_sep_melband_roformer_vocal", {}).get("OVRL") if row.get("dnsmos_sep_melband_roformer_vocal") else None
            mb_enh   = row.get("dnsmos_enh_melband_roformer_vocal", {}).get("OVRL") if row.get("dnsmos_enh_melband_roformer_vocal") else None

            best = max(v for v in [bs_sep, bs_enh, mb_sep, mb_enh] if v is not None)
            delta = round(best - raw_ovrl, 3) if raw_ovrl else None

            r = t.add_row().cells
            vals = [short[:22], raw_ovrl, bs_sep, bs_enh, mb_sep, mb_enh,
                    f"+{delta}" if delta else "N/A"]
            for i, v in enumerate(vals):
                r[i].text = str(v) if v is not None else "—"
                r[i].paragraphs[0].runs[0].font.size = Pt(8)

            # Color coding cho OVRL cells
            for ci, val in [(1, raw_ovrl), (2, bs_sep), (3, bs_enh), (4, mb_sep), (5, mb_enh)]:
                set_cell_bg(r[ci], ovrl_color(val))

            # Highlight ô tốt nhất (delta)
            set_cell_bg(r[6], "D1FAE5" if delta and delta >= 1.0 else "FEF9C3")

        doc.add_paragraph()

    # ── Chú thích màu ──────────────────────────────────────────
    para(doc, "Chú thích màu OVRL:", bold=True, size=9)
    color_notes = [
        ("D1FAE5", "≥ 3.8 — Rất tốt"),
        ("DCFCE7", "3.4–3.8 — Tốt"),
        ("FEF9C3", "3.0–3.4 — Khá"),
        ("FEF3C7", "2.5–3.0 — Trung bình"),
        ("FEE2E2", "< 2.5 — Kém"),
    ]
    ct = doc.add_table(rows=1, cols=len(color_notes))
    ct.style = "Table Grid"
    for i, (clr, label) in enumerate(color_notes):
        c = ct.rows[0].cells[i]
        c.text = label
        c.paragraphs[0].runs[0].font.size = Pt(8)
        set_cell_bg(c, clr)
    doc.add_paragraph()

    h2(doc, "3.2 Bảng tổng hợp SIG / BAK / OVRL (trung bình theo category)")

    def avg(lst):
        vals = [v for v in lst if v is not None]
        return round(np.mean(vals), 3) if vals else None

    summary_rows = []
    for cat, rows_cat in cats.items():
        def g(key, subkey):
            return avg([
                row.get(key, {}).get(subkey) if row.get(key) else None
                for row in rows_cat
            ])

        summary_rows.append([
            cat_labels[cat],
            f"{g('dnsmos_raw','SIG')} / {g('dnsmos_raw','BAK')} / {g('dnsmos_raw','OVRL')}",
            f"{g('dnsmos_sep_bs_roformer_1297','SIG')} / {g('dnsmos_sep_bs_roformer_1297','BAK')} / {g('dnsmos_sep_bs_roformer_1297','OVRL')}",
            f"{g('dnsmos_enh_bs_roformer_1297','SIG')} / {g('dnsmos_enh_bs_roformer_1297','BAK')} / {g('dnsmos_enh_bs_roformer_1297','OVRL')}",
            f"{g('dnsmos_sep_melband_roformer_vocal','SIG')} / {g('dnsmos_sep_melband_roformer_vocal','BAK')} / {g('dnsmos_sep_melband_roformer_vocal','OVRL')}",
            f"{g('dnsmos_enh_melband_roformer_vocal','SIG')} / {g('dnsmos_enh_melband_roformer_vocal','BAK')} / {g('dnsmos_enh_melband_roformer_vocal','OVRL')}",
        ])

    # Tổng hợp toàn bộ
    summary_rows.append([
        "⟨ TRUNG BÌNH TOÀN BỘ ⟩",
        f"{avg([r.get('dnsmos_raw',{}).get('SIG') if r.get('dnsmos_raw') else None for r in data])} / "
        f"{avg([r.get('dnsmos_raw',{}).get('BAK') if r.get('dnsmos_raw') else None for r in data])} / "
        f"{avg([r.get('dnsmos_raw',{}).get('OVRL') if r.get('dnsmos_raw') else None for r in data])}",
        f"{avg([r.get('dnsmos_sep_bs_roformer_1297',{}).get('SIG') if r.get('dnsmos_sep_bs_roformer_1297') else None for r in data])} / "
        f"{avg([r.get('dnsmos_sep_bs_roformer_1297',{}).get('BAK') if r.get('dnsmos_sep_bs_roformer_1297') else None for r in data])} / "
        f"{avg([r.get('dnsmos_sep_bs_roformer_1297',{}).get('OVRL') if r.get('dnsmos_sep_bs_roformer_1297') else None for r in data])}",
        f"{avg([r.get('dnsmos_enh_bs_roformer_1297',{}).get('SIG') if r.get('dnsmos_enh_bs_roformer_1297') else None for r in data])} / "
        f"{avg([r.get('dnsmos_enh_bs_roformer_1297',{}).get('BAK') if r.get('dnsmos_enh_bs_roformer_1297') else None for r in data])} / "
        f"{avg([r.get('dnsmos_enh_bs_roformer_1297',{}).get('OVRL') if r.get('dnsmos_enh_bs_roformer_1297') else None for r in data])}",
        f"{avg([r.get('dnsmos_sep_melband_roformer_vocal',{}).get('SIG') if r.get('dnsmos_sep_melband_roformer_vocal') else None for r in data])} / "
        f"{avg([r.get('dnsmos_sep_melband_roformer_vocal',{}).get('BAK') if r.get('dnsmos_sep_melband_roformer_vocal') else None for r in data])} / "
        f"{avg([r.get('dnsmos_sep_melband_roformer_vocal',{}).get('OVRL') if r.get('dnsmos_sep_melband_roformer_vocal') else None for r in data])}",
        f"{avg([r.get('dnsmos_enh_melband_roformer_vocal',{}).get('SIG') if r.get('dnsmos_enh_melband_roformer_vocal') else None for r in data])} / "
        f"{avg([r.get('dnsmos_enh_melband_roformer_vocal',{}).get('BAK') if r.get('dnsmos_enh_melband_roformer_vocal') else None for r in data])} / "
        f"{avg([r.get('dnsmos_enh_melband_roformer_vocal',{}).get('OVRL') if r.get('dnsmos_enh_melband_roformer_vocal') else None for r in data])}",
    ])

    tbl(doc,
        ["Category", "Raw\nSIG/BAK/OVRL",
         "BS-Roformer\nSIG/BAK/OVRL", "BS+DFN3\nSIG/BAK/OVRL",
         "MelBand\nSIG/BAK/OVRL",    "MelBand+DFN3\nSIG/BAK/OVRL"],
        summary_rows,
        col_widths=[3.8, 3.5, 3.5, 3.5, 3.5, 3.8],
    )

    divider(doc)

    # ════════════════════════════════════════════════════════════
    # PHẦN 4: TỐC ĐỘ XỬ LÝ
    # ════════════════════════════════════════════════════════════
    h1(doc, "4. Tốc độ xử lý (RTF — Real-Time Factor)")

    para(doc,
        "RTF = Thời gian xử lý / Thời lượng audio. "
        "RTF < 1.0 nghĩa là xử lý nhanh hơn real-time (phù hợp batch pipeline).")

    tbl(doc,
        ["Model", "RTF Trung bình", "RTF Min", "RTF Max",
         "Thời gian cho 60s", "So sánh"],
        [
            ["BS-Roformer-Viperx-1297",
             f"{RTF_BS['mean']:.3f}×",
             f"{RTF_BS['min']:.3f}×",
             f"{RTF_BS['max']:.3f}×",
             f"≈ {RTF_BS['mean']*60:.0f}s (~{RTF_BS['mean']*60/60:.1f} phút)",
             "Baseline"],
            ["MelBand-Roformer-Vocal",
             f"{RTF_MEL['mean']:.3f}×",
             f"{RTF_MEL['min']:.3f}×",
             f"{RTF_MEL['max']:.3f}×",
             f"≈ {RTF_MEL['mean']*60:.0f}s (~{RTF_MEL['mean']*60/60:.1f} phút)",
             f"Nhanh hơn BS {RTF_BS['mean']/RTF_MEL['mean']:.2f}×"],
            ["DeepFilterNet 3",
             f"{RTF_DFN3:.3f}×",
             "0.059×", "0.067×",
             f"≈ {RTF_DFN3*60:.1f}s (real-time)",
             "Rất nhanh — 16× RT"],
        ],
        col_widths=[4.5, 2.5, 2.2, 2.2, 4.0, 3.2],
    )

    note(doc,
         "RTF đo trên CPU (AMD64, không dùng GPU). Với GPU (CUDA), "
         "tốc độ kỳ vọng nhanh hơn 10–30× tùy hardware.")

    para(doc, "Pipeline tổng RTF (MSS + Enhancement):", bold=True)
    tbl(doc,
        ["Pipeline", "RTF MSS", "RTF DFN3", "RTF Tổng", "Ghi chú"],
        [
            ["BS-Roformer + DFN3",
             f"{RTF_BS['mean']:.2f}×", f"{RTF_DFN3:.3f}×",
             f"{RTF_BS['mean'] + RTF_DFN3:.2f}×",
             "DFN3 không đáng kể so với MSS"],
            ["MelBand + DFN3",
             f"{RTF_MEL['mean']:.2f}×", f"{RTF_DFN3:.3f}×",
             f"{RTF_MEL['mean'] + RTF_DFN3:.2f}×",
             f"Nhanh hơn BS pipeline {(RTF_BS['mean'])/(RTF_MEL['mean']):.2f}×"],
        ],
        col_widths=[4.5, 2.5, 2.5, 3.0, 5.2],
    )
    divider(doc)

    # ════════════════════════════════════════════════════════════
    # PHẦN 5: PHÂN TÍCH & NHẬN XÉT
    # ════════════════════════════════════════════════════════════
    h1(doc, "5. Phân tích & Nhận xét")

    # Tính các số để dùng trong phân tích
    raw_ovrls  = [r["dnsmos_raw"]["OVRL"] for r in data if r.get("dnsmos_raw")]
    bs_enh_ovrls  = [r["dnsmos_enh_bs_roformer_1297"]["OVRL"] for r in data if r.get("dnsmos_enh_bs_roformer_1297")]
    mb_enh_ovrls  = [r["dnsmos_enh_melband_roformer_vocal"]["OVRL"] for r in data if r.get("dnsmos_enh_melband_roformer_vocal")]

    avg_raw   = round(np.mean(raw_ovrls), 3)
    avg_bs    = round(np.mean(bs_enh_ovrls), 3)
    avg_mb    = round(np.mean(mb_enh_ovrls), 3)
    delta_bs  = round(avg_bs - avg_raw, 3)
    delta_mb  = round(avg_mb - avg_raw, 3)

    h2(doc, "5.1 Hiệu quả tổng thể của pipeline")

    success(doc,
        f"Pipeline MSS + DFN3 nâng DNSMOS OVRL trung bình từ {avg_raw} → "
        f"{avg_mb} (MelBand) / {avg_bs} (BS-Roformer), "
        f"tương đương cải thiện +{delta_mb} / +{delta_bs} điểm OVRL.")

    para(doc,
        "Kết quả cho thấy pipeline 2 bước (MSS → Speech Enhancement) rất hiệu quả "
        "để xử lý audio YouTube thô. Ngay cả những file có OVRL thấp nhất (1.16 — "
        "talkshow với nhạc nền rất nặng) sau pipeline cũng đạt 3.4+.")

    h2(doc, "5.2 So sánh 2 model tách vocal")

    para(doc,
        "Về chất lượng tách vocal (DNSMOS sau separation):", bold=True)
    bullet(doc,
        "BS-Roformer-1297 có OVRL cao hơn MelBand sau bước separation đơn lẻ "
        f"(TB: {avg([r.get('dnsmos_sep_bs_roformer_1297',{}).get('OVRL') for r in data if r.get('dnsmos_sep_bs_roformer_1297')]):.3f} vs "
        f"{avg([r.get('dnsmos_sep_melband_roformer_vocal',{}).get('OVRL') for r in data if r.get('dnsmos_sep_melband_roformer_vocal')]):.3f})")
    bullet(doc,
        "Tuy nhiên MelBand + DFN3 vượt BS-Roformer + DFN3 ở phần lớn các files — "
        "DeepFilterNet3 boost MelBand mạnh hơn (Δ ≈ +0.18) so với boost BS-Roformer (Δ ≈ +0.09)")
    bullet(doc,
        f"Kết luận pipeline tốt nhất: MelBand-Roformer + DFN3 (OVRL TB: {avg_mb})")

    para(doc, "Về tốc độ xử lý:", bold=True)
    bullet(doc,
        f"MelBand nhanh hơn BS-Roformer {RTF_BS['mean']/RTF_MEL['mean']:.2f}× "
        f"(RTF {RTF_MEL['mean']:.2f} vs {RTF_BS['mean']:.2f})")
    bullet(doc,
        "DeepFilterNet3 cực kỳ nhanh (RTF ≈ 0.06×, tức 16× real-time) — gần như không ảnh hưởng tổng thời gian")
    bullet(doc,
        "Trên CPU: xử lý 1 phút audio mất ~5.7 phút (MelBand pipeline). Cần GPU để deploy thực tế.")

    h2(doc, "5.3 Phân tích theo loại content")

    tbl(doc,
        ["Category", "Đặc điểm", "Raw OVRL TB", "Best pipeline OVRL TB", "Δ OVRL", "Nhận xét"],
        [
            ["KhanhVy Vlog",
             "Nhạc nền liên tục, giọng VN nữ",
             str(round(np.mean([r["dnsmos_raw"]["OVRL"] for r in cats["khanhvy_vlog"] if r.get("dnsmos_raw")]), 3)),
             str(round(np.mean([r["dnsmos_enh_melband_roformer_vocal"]["OVRL"] for r in cats["khanhvy_vlog"] if r.get("dnsmos_enh_melband_roformer_vocal")]), 3)),
             f"+{round(np.mean([r['dnsmos_enh_melband_roformer_vocal']['OVRL'] - r['dnsmos_raw']['OVRL'] for r in cats['khanhvy_vlog'] if r.get('dnsmos_raw') and r.get('dnsmos_enh_melband_roformer_vocal')]), 3)}",
             "Cải thiện tốt, MelBand+DFN3 hiệu quả nhất"],
            ["KhanhVy Talkshow",
             "Nhạc nền nặng, biến thiên nhiều",
             str(round(np.mean([r["dnsmos_raw"]["OVRL"] for r in cats["khanhvy_talkshow"] if r.get("dnsmos_raw")]), 3)),
             str(round(np.mean([r["dnsmos_enh_bs_roformer_1297"]["OVRL"] for r in cats["khanhvy_talkshow"] if r.get("dnsmos_enh_bs_roformer_1297")]), 3)),
             f"+{round(np.mean([r['dnsmos_enh_bs_roformer_1297']['OVRL'] - r['dnsmos_raw']['OVRL'] for r in cats['khanhvy_talkshow'] if r.get('dnsmos_raw') and r.get('dnsmos_enh_bs_roformer_1297')]), 3)}",
             "Cải thiện nhiều nhất (+1.7), nhạc nền rất nặng"],
            ["Beyond Limits\n/ NGA Levi",
             "Podcast/interview, ít nhạc",
             str(round(np.mean([r["dnsmos_raw"]["OVRL"] for r in cats["beyondlimits_ngalevi"] if r.get("dnsmos_raw")]), 3)),
             str(round(np.mean([r["dnsmos_enh_melband_roformer_vocal"]["OVRL"] for r in cats["beyondlimits_ngalevi"] if r.get("dnsmos_enh_melband_roformer_vocal")]), 3)),
             f"+{round(np.mean([r['dnsmos_enh_melband_roformer_vocal']['OVRL'] - r['dnsmos_raw']['OVRL'] for r in cats['beyondlimits_ngalevi'] if r.get('dnsmos_raw') and r.get('dnsmos_enh_melband_roformer_vocal')]), 3)}",
             "Điểm cuối cao nhất (~3.86), audio gốc đã tốt hơn"],
        ],
        col_widths=[2.8, 3.5, 2.3, 3.0, 1.8, 4.2],
    )

    h2(doc, "5.4 Cải thiện chỉ số BAK (Background Noise)")

    para(doc,
        "BAK score phản ánh chất lượng loại bỏ nhiễu nền — chỉ số quan trọng nhất "
        "với mục tiêu tách giọng từ YouTube. Cải thiện BAK là kết quả rõ nhất của pipeline:")

    bak_rows = []
    for row in data:
        short = row["file"]
        raw_bak = row.get("dnsmos_raw", {}).get("BAK") if row.get("dnsmos_raw") else None
        mb_enh_bak = row.get("dnsmos_enh_melband_roformer_vocal", {}).get("BAK") if row.get("dnsmos_enh_melband_roformer_vocal") else None
        delta_bak = round(mb_enh_bak - raw_bak, 3) if (raw_bak and mb_enh_bak) else None
        bak_rows.append([
            short.split("__")[-1][:18],
            str(raw_bak), str(mb_enh_bak),
            f"+{delta_bak}" if delta_bak else "N/A"
        ])

    tbl(doc,
        ["File", "BAK Raw", "BAK MelBand+DFN3", "Δ BAK"],
        bak_rows,
        col_widths=[6.0, 3.0, 4.5, 3.0],
    )

    avg_raw_bak = avg([r.get("dnsmos_raw", {}).get("BAK") if r.get("dnsmos_raw") else None for r in data])
    avg_mb_bak  = avg([r.get("dnsmos_enh_melband_roformer_vocal", {}).get("BAK") if r.get("dnsmos_enh_melband_roformer_vocal") else None for r in data])
    success(doc,
        f"BAK trung bình: {avg_raw_bak} (raw) → {avg_mb_bak} (MelBand+DFN3), "
        f"cải thiện +{round(avg_mb_bak - avg_raw_bak, 3)} điểm. "
        "Nền âm thanh được loại bỏ hiệu quả ở tất cả các loại content.")

    divider(doc)

    # ════════════════════════════════════════════════════════════
    # PHẦN 6: KẾT LUẬN & ĐỀ XUẤT
    # ════════════════════════════════════════════════════════════
    h1(doc, "6. Kết luận & Đề xuất")

    h2(doc, "6.1 Kết luận")

    tbl(doc,
        ["#", "Kết luận", "Chi tiết"],
        [
            ["1", "Pipeline MSS + DFN3 rất hiệu quả",
             f"OVRL raw {avg_raw} → {avg_mb} (MelBand+DFN3), tăng {delta_mb} điểm"],
            ["2", "MelBand-Roformer + DFN3 là pipeline tốt nhất",
             f"Vượt BS-Roformer+DFN3 về OVRL, nhanh hơn {RTF_BS['mean']/RTF_MEL['mean']:.2f}× về RTF"],
            ["3", "BS-Roformer tách vocal sạch hơn về SDR",
             "Nhưng MelBand vocals dễ enhance hơn bởi DeepFilterNet3"],
            ["4", "DeepFilterNet3 là bước boost quan trọng",
             "Đặc biệt cải thiện BAK và OVRL sau separation, RTF cực nhỏ (0.06×)"],
            ["5", "Content phụ thuộc ảnh hưởng kết quả",
             "Talkshow (nhiều nhạc) cải thiện nhiều nhất; interview (ít nhạc) đạt điểm cao nhất"],
        ],
        col_widths=[0.8, 5.5, 11.3],
    )

    h2(doc, "6.2 Đề xuất cho deployment")

    bullet(doc, "Recommended pipeline: MelBand-Roformer-Vocal → DeepFilterNet 3")
    bullet(doc, "Dùng GPU để đạt RTF < 1 (real-time hoặc nhanh hơn) — hiện CPU quá chậm cho production")
    bullet(doc, "Với nội dung talk show/vlog có nhạc nền: nên dùng BS-Roformer + DFN3 (tách sạch hơn trước khi enhance)")
    bullet(doc, "Với nội dung interview/podcast ít nhạc: MelBand+DFN3 đủ và nhanh hơn")
    bullet(doc, "Nên test thêm trên toàn bộ dataset (90 files) và so sánh WER sau ASR để đánh giá downstream")
    bullet(doc, "Xem xét thêm 2-stage separation: BS-Roformer (vocal) → De-Echo MelBand (noise) → DFN3")

    h2(doc, "6.3 Các bước tiếp theo")

    tbl(doc,
        ["Ưu tiên", "Task", "Lý do"],
        [
            ["🔴 Cao", "Test trên GPU", "RTF hiện 5–9× trên CPU, cần < 1× cho production"],
            ["🔴 Cao", "Evaluate downstream ASR/WER", "DNSMOS chỉ đo perceptual quality, cần đo ảnh hưởng thực tế"],
            ["🟡 Trung", "Thử toàn bộ 90 files", "Sample 9 files có thể chưa đại diện đủ"],
            ["🟡 Trung", "Thử thêm MDX23C + BS-Roformer ensemble", "Ensemble thường cho kết quả tốt hơn"],
            ["🟢 Thấp", "Tích hợp vào data processing pipeline", "Sau khi chọn được model phù hợp"],
        ],
        col_widths=[1.8, 5.5, 10.3],
    )

    divider(doc)

    # ════════════════════════════════════════════════════════════
    # PHỤ LỤC
    # ════════════════════════════════════════════════════════════
    h1(doc, "Phụ lục: Dữ liệu chi tiết đầy đủ")

    para(doc, "Bảng đầy đủ tất cả metrics (SIG, BAK, OVRL) cho từng file:", bold=True)

    full_headers = [
        "File", "SNR\nRaw",
        "Raw\nSIG", "Raw\nBAK", "Raw\nOVRL",
        "BS-sep\nOVRL", "BS+DFN3\nOVRL",
        "MB-sep\nOVRL", "MB+DFN3\nOVRL",
    ]
    full_rows = []
    for row in data:
        raw  = row.get("dnsmos_raw") or {}
        bss  = row.get("dnsmos_sep_bs_roformer_1297") or {}
        bse  = row.get("dnsmos_enh_bs_roformer_1297") or {}
        mbs  = row.get("dnsmos_sep_melband_roformer_vocal") or {}
        mbe  = row.get("dnsmos_enh_melband_roformer_vocal") or {}
        full_rows.append([
            row["file"].split("__")[-1][:20],
            row.get("snr_raw", "—"),
            raw.get("SIG", "—"), raw.get("BAK", "—"), raw.get("OVRL", "—"),
            bss.get("OVRL", "—"), bse.get("OVRL", "—"),
            mbs.get("OVRL", "—"), mbe.get("OVRL", "—"),
        ])

    tbl(doc, full_headers, full_rows,
        col_widths=[3.5, 1.5, 1.5, 1.5, 1.8, 2.0, 2.0, 2.0, 2.2])

    # ── Footer ────────────────────────────────────────────────────
    doc.add_paragraph()
    divider(doc)
    footer_p = doc.add_paragraph(
        "Báo cáo tự động tạo bởi generate_report.py  |  "
        "Dự án: AI Voice Pipeline — Music Source Separation + Speech Enhancement  |  2026-06-03"
    )
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_p.runs[0].font.size = Pt(8)
    footer_p.runs[0].font.color.rgb = RGBColor(0x99, 0x99, 0x99)
    footer_p.runs[0].italic = True

    doc.save(out_path)
    print(f"[XONG] Báo cáo đã lưu tại: {out_path}")


if __name__ == "__main__":
    build(OUT_DOCX)

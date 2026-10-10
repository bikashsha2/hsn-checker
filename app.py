import io
import os
import re
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber
import pypdfium2 as pdfium
import pytesseract
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
HSN_CSV = BASE_DIR / "hsn_master.csv"
BACKUP_DIR = BASE_DIR / "hsn_backups"
CHANGELOG_CSV = BASE_DIR / "hsn_changelog.csv"

# Use the default Windows install location only if it actually exists,
# otherwise rely on Tesseract being on PATH (Linux / macOS / custom installs).
_WIN_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(_WIN_TESSERACT):
    pytesseract.pytesseract.tesseract_cmd = _WIN_TESSERACT


@st.cache_resource(show_spinner=False)
def tesseract_available():
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


TESSERACT_MISSING_MSG = (
    "❌ Tesseract OCR is not installed (or not found). It is required to read scanned PDFs "
    "and images. [Download Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki) "
    "and install it to the default location, then restart the app."
)

SKIP_PAGE_KEYWORDS = ["terms and conditions", "terms & conditions", "declaration"]

st.set_page_config(page_title="HSN Checker", page_icon="✅", layout="centered", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
      #MainMenu, footer, .stDeployButton, [data-testid="stAppDeployButton"] {display: none;}
      .block-container {padding-top: 2.5rem; max-width: 900px;}
      [data-testid="stMetricValue"] {font-size: 1.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("HSN Code Checker")
st.caption("Check invoice HSN/RITC codes against the DGFT ITC(HS) list.")


# ---------------------------------------------------------------------------
# Master database
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading Master HSN Database...")
def load_master_hsn_db():
    """Returns (dataframe, error_message). Codes are kept as strings to preserve leading zeros."""
    if not HSN_CSV.exists():
        return pd.DataFrame(), f"File not found: {HSN_CSV}"
    try:
        df = pd.read_csv(HSN_CSV, dtype=str).fillna("")
    except Exception as e:
        return pd.DataFrame(), f"Could not read {HSN_CSV.name}: {e}"
    missing = {"HSN Code", "Description"} - set(df.columns)
    if missing:
        return pd.DataFrame(), f"{HSN_CSV.name} is missing column(s): {', '.join(sorted(missing))}"
    if "Policy" not in df.columns:
        df["Policy"] = "Unknown"
    if "Status" not in df.columns:
        df["Status"] = "Active"
    if "Updated" not in df.columns:
        df["Updated"] = ""
    df.loc[df["Status"].str.strip() == "", "Status"] = "Active"
    df["HSN Code"] = df["HSN Code"].str.strip()
    df = df[df["HSN Code"] != ""].drop_duplicates(subset="HSN Code")
    return df, None


hsn_db, db_error = load_master_hsn_db()
if db_error:
    st.error(f"❌ HSN Database could not be loaded. {db_error}")
    st.stop()

HSN_LOOKUP = hsn_db.set_index("HSN Code")[["Description", "Policy", "Status"]].to_dict("index")
VALID_CHAPTERS = {c[:2] for c in HSN_LOOKUP}
VALID_PREFIXES = {c[:n] for c in HSN_LOOKUP for n in (4, 6)}
HAS_POLICY_DATA = any(str(v["Policy"]).strip().lower() not in ("", "unknown") for v in HSN_LOOKUP.values())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_skippable_page(text):
    """Terms & conditions / declaration-only pages that contain no HSN data."""
    t = (text or "").lower()
    return any(k in t for k in SKIP_PAGE_KEYWORDS) and not any(k in t for k in ("hsn", "ritc"))


def is_summary_header(text):
    """The 'HSN/SAC | Taxable Value | CGST ...' tax summary at the bottom of GST invoices.
    It repeats codes already listed in the item table, so it must not be counted again."""
    t = (text or "").lower()
    if not ("hsn" in t or "ritc" in t):
        return False
    if "summary" in t:
        return True
    # e.g. "HSN | Qty. In Kg. | FOB before Disc | Net value | Generic description"
    words = t.split()
    if words and re.match(r"^\W*(hsn|ritc)", words[0]) and any(k in t for k in ("taxable", "net value", "generic")) \
            and not any(k in t for k in ("rate", "weight", "unit price", "particular")):
        return True
    item_words = ("qty", "quantity", "rate", "description", "particular", "goods", "item", "unit")
    return "taxable" in t and not any(w in t for w in item_words)


# Invoices are often printed as Original / Duplicate / Triplicate copies in one PDF.
COPY_RE = re.compile(r"\b(original|duplicate|triplicate|quadruplicate)\s+(?:copy|for)\b", re.I)


def copy_label(text):
    m = COPY_RE.search(text or "")
    return m.group(1).lower() if m else None


OCR_SCALE = 3.0  # render scale for OCR (3x ~ 216 dpi); small table text needs this


def _count_valid(items):
    return sum(1 for item in items if item["Code"] in HSN_LOOKUP)


def _ocr_words(image, config):
    d = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    return [dict(text=d["text"][i].strip(), x=d["left"][i], y=d["top"][i], w=d["width"][i], h=d["height"][i],
                 line=(d["block_num"][i], d["par_num"][i], d["line_num"][i]))
            for i in range(len(d["text"])) if d["text"][i].strip()]


def _vertical_lines(gray, y0, y1):
    """x positions of the table's vertical rule lines between rows y0..y1."""
    arr = np.asarray(gray, dtype=np.uint8)[y0:y1]
    if arr.size == 0:
        return []
    xs = np.where((arr < 128).mean(axis=0) >= 0.5)[0]
    lines, start, prev = [], None, None
    for x in xs:
        if start is None or x != prev + 1:
            if start is not None:
                lines.append(int((start + prev) // 2))
            start = x
        prev = x
    if start is not None:
        lines.append(int((start + prev) // 2))
    return lines


def _column_box(gray, header, y_end):
    W = gray.size[0]
    y0 = header["y"] + header["h"] + 5
    lines = _vertical_lines(gray, y0, max(y0 + 10, y_end))
    left = max([x for x in lines if x < header["x"]], default=None)
    right = min([x for x in lines if x > header["x"] + header["w"]], default=None)
    if left is None or right is None or right - left > W * 0.3:
        # no ruled lines: assume a column a few times the header width, centred on it
        cx, half = header["x"] + header["w"] // 2, max(header["w"] * 2, int(W * 0.06))
        left, right = max(0, cx - half), min(W, cx + half)
    return left + 3, right - 3


def _clean_crop(crop, pad=30, vertical=True):
    """Erase rule lines and add a white margin: Tesseract tends to stop reading a
    column part-way down when row lines touch the text."""
    a = np.asarray(crop, dtype=np.uint8).copy()
    if a.size:
        a[(a < 128).mean(axis=1) > 0.5, :] = 255
        if vertical:  # off for short cells, where a "1" or "L" stroke looks like a rule line
            a[:, (a < 128).mean(axis=0) > 0.5] = 255
    return ImageOps.expand(Image.fromarray(a), border=pad, fill=255), pad


def _words_to_text(words, keep=None):
    lines, order = {}, []
    for w in words:
        if w["line"] not in lines:
            order.append(w["line"])
        lines.setdefault(w["line"], []).append(w)
    out, prev_block = [], None
    for k in order:
        if keep is not None and not keep(lines[k]):
            continue
        if prev_block is not None and k[0] != prev_block:
            out.append("")
        out.append(" ".join(w["text"] for w in lines[k]))
        prev_block = k[0]
    return "\n".join(out)


def ocr_page_text(gray):
    """One --psm 3 pass. Returns (text, words, search_text).
    `search_text` drops everything physically below the HSN tax-summary header, so the
    summary is never counted even when Tesseract outputs its block before the items."""
    words = _ocr_words(gray, "--psm 3")
    by_line = {}
    for w in words:
        by_line.setdefault(w["line"], []).append(w)
    summary_tops = [min(w["y"] for w in ws) for ws in by_line.values()
                    if is_summary_header(" ".join(w["text"] for w in ws))]
    cutoff = min(summary_tops) if summary_tops else None
    text = _words_to_text(words)
    search_text = _words_to_text(words, keep=lambda ws: cutoff is None or min(w["y"] for w in ws) < cutoff - 5)
    return text, words, search_text


def attach_ocr_descriptions(items, words):
    """Fill each item's description from the words to the LEFT of its HSN code, between
    this item's row and the next one (handles descriptions wrapped over 2-3 lines)."""
    if not items or not words:
        return
    ordered = sorted(words, key=lambda w: (w["y"], w["x"]))
    ptr = 0
    for it in items:  # locate codes found in the text on the page
        if "_top" in it:
            continue
        for k in range(ptr, len(ordered)):
            w = ordered[k]
            if re.sub(r"\D", "", _fix_ocr_token(w["text"])) == it["Code"]:
                it.update(_top=w["y"], _h=w["h"], _x=w["x"])
                ptr = k + 1
                break
    located = sorted([it for it in items if "_top" in it], key=lambda it: it["_top"])
    for n, it in enumerate(located):
        if it.get("Description") and "_x" in it and it.get("_from_cells"):
            continue
        top = it["_top"] - it["_h"] * 0.6
        if n + 1 < len(located):
            bottom = located[n + 1]["_top"] - located[n + 1]["_h"] * 0.6
        else:
            bottom = it["_top"] + it["_h"] * 2.5
        in_band = [w for w in words if top <= w["y"] < bottom and w["x"] + w["w"] <= it["_x"] - 5]
        lines = {}
        for w in in_band:
            lines.setdefault(w["line"], []).append(w)
        parts = []
        for ws in sorted(lines.values(), key=lambda ws: min(w["y"] for w in ws)):
            toks = [w["text"] for w in sorted(ws, key=lambda w: w["x"])]
            if any(re.fullmatch(r"(sub|grand)?total:?", t, re.I) for t in toks):
                break
            while toks and not re.search(r"[A-Za-z0-9]", toks[0]):
                toks = toks[1:]  # specks / punctuation
            if toks and re.fullmatch(r"\W*\d{1,3}\W*", toks[0]):
                toks = toks[1:]  # Sl. No.
            while toks and not re.search(r"[A-Za-z0-9]", toks[-1]):
                toks = toks[:-1]
            parts.append(" ".join(toks))
        desc = clean_desc(" ".join(parts))
        if desc:
            it["Description"] = desc


def ocr_table_columns(gray, words=None):
    """Column-based OCR for dense table invoices: find the 'HSN' header, crop just that
    column (down to the 'Total' row / tax summary), OCR it alone, and pair each code
    with the Sl./Sr. No. on the same row."""
    if words is None or not any(re.match(r"^\W*(hsn|ritc)", w["text"], re.I) for w in words):
        words = _ocr_words(gray, "--psm 11")  # sparse-text mode finds headers psm 3 can miss
    headers = sorted([w for w in words if re.match(r"^\W*(hsn|ritc)", w["text"], re.I)], key=lambda w: w["y"])
    if not headers:
        return []
    hdr = headers[0]
    hdr_line = " ".join(w["text"] for w in words if w["line"] == hdr["line"])
    if is_summary_header(hdr_line):
        return []  # first HSN header on this page is the tax summary, not the item table
    ends = [w["y"] for w in headers[1:] if w["y"] > hdr["y"] + 30]
    ends += [w["y"] for w in words if re.fullmatch(r"(sub)?-?total:?", w["text"], re.I)
             and w["x"] < hdr["x"] and w["y"] > hdr["y"] + 30]
    y_end = min(ends) if ends else gray.size[1]
    y0 = hdr["y"] + hdr["h"]
    if y_end - y0 < 20:
        return []

    x0, x1 = _column_box(gray, hdr, y_end)
    crop, pad = _clean_crop(gray.crop((x0, y0, x1, y_end)))
    lines = {}
    for w in _ocr_words(crop, "--psm 6 -c tessedit_char_whitelist=0123456789."):
        lines.setdefault(w["line"], []).append(w)
    codes = []
    for toks in lines.values():
        digits = "".join(t["text"] for t in sorted(toks, key=lambda t: t["x"])).replace(".", "")
        if digits.isdigit() and len(digits) in (4, 6, 8) and digits[:2] in VALID_CHAPTERS:
            top = min(t["y"] for t in toks) - pad
            bottom = max(t["y"] + t["h"] for t in toks) - pad
            codes.append((digits, y0 + top, y0 + bottom))

    sls = []
    sl_hdr = [w for w in words if re.fullmatch(r"(sr|sl|s)[.,]?(no\.?)?", w["text"], re.I)
              and abs(w["y"] - hdr["y"]) < hdr["h"] * 3 and w["x"] < hdr["x"]]
    if sl_hdr:
        sx0, sx1 = _column_box(gray, sl_hdr[0], y_end)
        if sx1 > sx0:
            scrop, spad = _clean_crop(gray.crop((sx0, y0, sx1, y_end)))
            for w in _ocr_words(scrop, "--psm 6 -c tessedit_char_whitelist=0123456789"):
                if re.fullmatch(r"\d{1,4}", w["text"]):
                    sls.append((w["text"].lstrip("0") or "0", y0 + w["y"] - spad + w["h"] / 2))

    items = []
    for code, top, bottom in codes:
        mid = (top + bottom) / 2
        best = min(sls, key=lambda s: abs(s[1] - mid), default=None)
        sl = best[0] if best and abs(best[1] - mid) < (bottom - top) * 1.5 else ""
        items.append({"Sl. No.": sl, "Code": code, "Description": "",
                      "_top": top, "_h": bottom - top, "_x": x0})

    # Item descriptions: OCR each row's cell in the Description column, using the
    # table's horizontal row lines (measured in the HSN column) as row boundaries.
    try:
        _ocr_description_cells(gray, words, hdr, items, x0, x1, y0, y_end)
    except Exception:
        pass
    return items


def _horizontal_lines(gray, x0, x1, y0, y1):
    arr = np.asarray(gray.crop((x0, y0, x1, y1)), dtype=np.uint8)
    if arr.size == 0:
        return []
    ys = np.where((arr < 128).mean(axis=1) > 0.5)[0]
    lines, start, prev = [], None, None
    for y in ys:
        if start is None or y != prev + 1:
            if start is not None:
                lines.append(int(y0 + (start + prev) // 2))
            start = y
        prev = y
    if start is not None:
        lines.append(int(y0 + (start + prev) // 2))
    return lines


def _ocr_description_cells(gray, words, hdr, items, x0, x1, y0, y_end):
    if not items:
        return
    rows = [y0] + _horizontal_lines(gray, x0, x1, y0, y_end) + [y_end]
    if len(rows) < 3:
        return  # no ruled rows: positional fallback is used instead
    cands = [w for w in words if DESC_HEADER_RE.search(w["text"]) and w["x"] + w["w"] < hdr["x"]
             and abs(w["y"] - hdr["y"]) < hdr["h"] * 3]
    if not cands:
        return
    dh = sorted(cands, key=lambda w: (0 if "desc" in w["text"].lower() else 1, -w["x"]))[0]
    dx0, dx1 = _column_box(gray, dh, y_end)
    if dx1 - dx0 < 20 or dx1 > x0 + 10:
        return
    for it in items:
        mid = it["_top"] + it["_h"] / 2
        top = max([r for r in rows if r < mid], default=None)
        bottom = min([r for r in rows if r > mid], default=None)
        if top is None or bottom is None or bottom - top < 10:
            continue
        cell, _ = _clean_crop(gray.crop((dx0, top + 3, dx1, bottom - 2)), pad=15, vertical=False)
        text = pytesseract.image_to_string(cell, config="--psm 6")
        desc = clean_desc(" ".join(text.split()))
        if desc:
            it["Description"] = desc
            it["_from_cells"] = True


def ocr_page_items(gray, text, words, search_text=None):
    """Find HSN codes on an OCR'd page, trying several readings and keeping the one with
    the most valid codes:
    - the --psm 3 text (automatic layout): most scans
    - column OCR: dense ruled tables where full-page OCR drops or misreads rows
    - --psm 6 (one uniform block): grid-heavy scans where psm 3 skips whole columns
    Returns (text, items)."""
    best = find_codes_in_text(search_text if search_text is not None else text)
    best_words = words
    try:
        col_items = ocr_table_columns(gray, words)
    except Exception:
        col_items = []
    if col_items and _count_valid(col_items) >= _count_valid(best):
        best = col_items

    if _count_valid(best) == 0:
        words6 = _ocr_words(gray, "--psm 6")
        text6 = _words_to_text(words6)
        items6 = find_codes_in_text(text6)
        if _count_valid(items6) > 0:
            text, best, best_words = text6, items6, words6

    try:
        attach_ocr_descriptions(best, best_words)
    except Exception:
        pass
    return text, best


def ocr_page(image):
    gray = image.convert("L")
    text, words, search_text = ocr_page_text(gray)
    if is_skippable_page(text):
        return text, []
    return ocr_page_items(gray, text, words, search_text)


# OCR often reads 0 as O, 1 as I/l, 5 as S. Only fix this inside tokens that are
# already mostly digits, so ordinary words ("sit", "oe") never turn into numbers.
_OCR_DIGIT_FIX = str.maketrans("OoIlisS", "0011155")


def _fix_ocr_token(tok):
    fixed = tok.translate(_OCR_DIGIT_FIX)
    if fixed != tok and sum(ch.isdigit() for ch in tok) >= 5 and re.fullmatch(r"[\d.]+", fixed):
        return fixed
    return tok


# An 8-digit code, optionally written as 1234 5678 / 1234.56.78 / 123456 78.
# It must not touch other digits, commas or decimals (so amounts like 2,568 or
# separate columns like "1999 205" are never glued together).
CODE_RE = re.compile(r"(?<![\d,.])(\d{4}(?:[ .]?\d{2}){2})(?![\d]|[.,]\d)")
AMOUNT_RE = re.compile(r"\d,\d{2,3}\b|\d\.\d{2}\b")
SL_RE = re.compile(r"^0*(\d{1,3})[.)\-]?$")


def find_codes_in_text(text):
    """Text-based fallback (used for scanned pages and PDFs without real tables).
    Returns a list of {'Sl. No.', 'Code'}; Sl. No. may be '' when the line has none."""
    results = []
    for raw_line in (text or "").split("\n"):
        if is_summary_header(raw_line):
            break  # everything below is the HSN tax summary / totals / bank details

        tokens = [_fix_ocr_token(t) for t in raw_line.strip().split()]
        line = " ".join(tokens)
        line_has_amount = bool(AMOUNT_RE.search(line))
        line_hits = []

        for m in CODE_RE.finditer(line):
            code = re.sub(r"[ .]", "", m.group(1))
            if code[:2] not in VALID_CHAPTERS:
                continue
            # Position of the code in the line (in tokens)
            idx = len(line[: m.start()].split())
            starts_with_sl = idx == 1 and SL_RE.match(tokens[0] or "")
            # A real HSN cell is the first thing on the line (OCR column layout), right
            # after the Sl. No., or on an item row that has amounts. A number buried in
            # a description ("CURTAIN - 63031100 75\" READY ...") is ignored.
            if not (idx == 0 or starts_with_sl or line_has_amount):
                continue
            sl = SL_RE.match(tokens[0]).group(1) if tokens and SL_RE.match(tokens[0]) and idx > 0 else ""
            # description = words between the Sl. No. and the code on the same line
            desc = clean_desc(" ".join(t for t in tokens[(1 if sl else 0):idx] if not CODE_RE.fullmatch(t)))
            line_hits.append({"Sl. No.": sl, "Code": code, "Description": desc})

        # An item row has exactly one HSN cell and it comes after the description, so if
        # a line has several candidates (e.g. "CURTAIN - 63031100 ... 63039990 187 PCS")
        # only the last one is the real HSN code.
        if line_hits:
            results.append(line_hits[-1])
    return results


DESC_HEADER_RE = re.compile(r"desc|particular|item|goods|product|material|commodity|article", re.I)


def clean_desc(text):
    text = re.sub(r"[|\[\]{}‘’“”_~]+", " ", str(text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[^A-Za-z0-9(]+|[^A-Za-z0-9)\"']+$", "", text)
    return text[:150]


def _find_desc_col(lower_cells, exclude):
    """Index of the item-description column in a header row (prefers 'description')."""
    cands = [k for k, c in enumerate(lower_cells) if k not in exclude and c and DESC_HEADER_RE.search(c)
             and not SL_HEADER_RE.search(c) and "hsn" not in c and "code" != c.strip()]
    pref = [k for k in cands if "desc" in lower_cells[k] or "particular" in lower_cells[k]]
    return (pref or cands or [-1])[0]


def _norm_cell(cell):
    return str(cell or "").lower().replace("\n", " ").strip()


SL_HEADER_RE = re.compile(r"^(sl|sr|s)\.?\s*(no|n0|#)?\.?$|^(sl|sr|s)\.?\s*no\b|^#$|^no\.?$")


def find_codes_in_tables(page, state):
    """Reads the HSN column of real PDF tables. `state` carries the header layout over
    to continuation pages whose tables have no header row."""
    results = []
    for table in page.extract_tables() or []:
        if not table or len(table) < 1:
            continue
        hsn_col = sl_col = desc_col = header_row = -1
        is_summary = False

        for r_idx, row in enumerate(table[:10]):
            if not row:
                continue
            cells = [_norm_cell(c) for c in row]
            row_text = " ".join(cells)
            for c_idx, cell in enumerate(cells):
                if hsn_col == -1 and ("hsn" in cell or "ritc" in cell):
                    hsn_col = c_idx
                if sl_col == -1 and SL_HEADER_RE.search(cell):
                    sl_col = c_idx
            if hsn_col != -1:
                header_row = r_idx
                is_summary = is_summary_header(row_text)
                desc_col = _find_desc_col(cells, {hsn_col, sl_col})
                break

        if hsn_col != -1:
            if is_summary:
                continue  # skip the HSN tax-summary table
            state.update(hsn_col=hsn_col, sl_col=sl_col, desc_col=desc_col, ncols=len(table[header_row]))
        elif state.get("hsn_col", -1) != -1 and len(table[0] or []) == state["ncols"]:
            # Continuation table on a later page with the same column layout
            hsn_col, sl_col, desc_col, header_row = state["hsn_col"], state["sl_col"], state["desc_col"], -1
        else:
            continue

        for row in table[header_row + 1:]:
            if not row or len(row) <= hsn_col or not row[hsn_col]:
                continue
            code = re.sub(r"[.\s\-]", "", str(row[hsn_col]))
            if not code.isdigit() or len(code) not in (4, 6, 8):
                continue
            sl = ""
            if sl_col != -1 and len(row) > sl_col and row[sl_col]:
                m = re.match(r"\s*0*(\d{1,4})", str(row[sl_col]))
                sl = m.group(1) if m else ""
            desc = clean_desc(row[desc_col]) if desc_col != -1 and len(row) > desc_col else ""
            results.append({"Sl. No.": sl, "Code": code, "Description": desc})
    return results


# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------
def process_pdf(pdf_bytes, force_ocr=False):
    """Returns (raw_text, items, notes, all_pages_ocred)."""
    text_parts, items, notes = [], [], []
    first_copy, skipped_copy_pages = None, []
    pages_total = pages_ocred = 0
    ocr_ok = tesseract_available()
    ocr_needed_but_missing = False
    table_state = {}

    pdf_doc = pdfium.PdfDocument(pdf_bytes)  # opened once, reused for every page
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                pages_total += 1
                native_text = "" if force_ocr else (page.extract_text() or "")
                if native_text and is_skippable_page(native_text):
                    continue

                # 1. Real tables (native / digital PDFs)
                page_items = [] if force_ocr else find_codes_in_tables(page, table_state)

                # 2. Little or no text layer -> scanned page -> OCR
                page_text, ocr_items, ocr_state = native_text, None, None
                if force_ocr or len(native_text.split()) < 50:
                    if ocr_ok:
                        try:
                            gray = pdf_doc[i].render(scale=OCR_SCALE).to_pil().convert("L")
                            ocr_text, words, search_text = ocr_page_text(gray)
                            pages_ocred += 1
                            if is_skippable_page(ocr_text):
                                continue
                            if len(ocr_text.split()) > len(native_text.split()):
                                page_text = ocr_text
                            ocr_state = (gray, ocr_text, words, search_text)
                        except Exception as e:
                            notes.append(f"OCR failed on page {i + 1}: {e}")
                    else:
                        ocr_needed_but_missing = True

                # 3. Duplicate / Triplicate copies of the same invoice: count items once
                label = copy_label(page_text)
                if label and first_copy is None:
                    first_copy = label
                elif label and label != first_copy and items:
                    # the rest of the file is the next copy (Duplicate/Triplicate...)
                    skipped_copy_pages = list(range(i + 1, len(pdf.pages) + 1))
                    pages_ocred = pages_total = 1  # don't trigger the full-OCR retry
                    break

                text_parts.append(f"----- Page {i + 1} -----\n{page_text}")

                # 4. Text search for pages where no table data was found
                if not page_items and ocr_state is not None:
                    try:
                        ocr_text, page_items = ocr_page_items(*ocr_state)
                        page_text = ocr_text if len(ocr_text.split()) > len(native_text.split()) else page_text
                        text_parts[-1] = f"----- Page {i + 1} -----\n{page_text}"
                    except Exception as e:
                        notes.append(f"OCR failed on page {i + 1}: {e}")
                if not page_items and ocr_state is None:
                    page_items = find_codes_in_text(page_text)
                items.extend(page_items)
    finally:
        pdf_doc.close()

    if skipped_copy_pages:
        pages = f"{skipped_copy_pages[0]}–{skipped_copy_pages[-1]}" if len(skipped_copy_pages) > 1 else str(skipped_copy_pages[0])
        notes.append(f"ℹ️ Page(s) {pages} are Duplicate/Triplicate copies of the same invoice and were not counted again.")
    if ocr_needed_but_missing:
        notes.append("TESSERACT_MISSING")
    return "\n".join(text_parts), items, notes, pages_total > 0 and pages_ocred == pages_total


def _deskew(gray):
    """Straighten a slightly tilted photo (up to ±6°) by finding the angle at which the
    text rows line up best (sharpest horizontal projection profile)."""
    small = gray.copy()
    small.thumbnail((1000, 1000))
    ink = Image.fromarray(((np.asarray(small) < 128) * 255).astype(np.uint8))
    best_score, best_angle = -1.0, 0.0
    for angle in np.arange(-6, 6.01, 0.25):
        rows = (np.asarray(ink.rotate(float(angle), resample=Image.NEAREST, fillcolor=0)) > 0).sum(axis=1)
        score = float(np.var(rows))
        if score > best_score:
            best_score, best_angle = score, float(angle)
    if abs(best_angle) < 0.3:
        return gray
    return gray.rotate(best_angle, resample=Image.BICUBIC, expand=True, fillcolor=255)


def prepare_photo(image, target=3000):
    """Phone/webcam photo -> clean grayscale page for OCR."""
    image = ImageOps.exif_transpose(image)  # apply the camera's rotation
    gray = ImageOps.autocontrast(image.convert("L"), cutoff=1)
    if max(gray.size) < target:  # small photos/screenshots: upscale for better OCR
        f = target / max(gray.size)
        gray = gray.resize((int(gray.width * f), int(gray.height * f)), Image.LANCZOS)
    return _deskew(gray)


def process_image(image_bytes):
    if not tesseract_available():
        return "", [], ["TESSERACT_MISSING"]
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        text, items = ocr_page(prepare_photo(image))
    except UnidentifiedImageError:
        return "", [], ["The uploaded file is not a valid image or is corrupted."]
    except Exception as e:
        return "", [], [f"Could not read the image: {e}"]
    return text, items, []


# ---------------------------------------------------------------------------
# Excel / CSV invoices
# ---------------------------------------------------------------------------
def _cell_str(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def _excel_code(v):
    """Normalise an HSN cell. Excel stores codes typed as numbers without their leading
    zero (07020000 -> 7020000), so a 3/5/7-digit numeric value gets its zero back."""
    numeric = isinstance(v, (int, float)) and not isinstance(v, bool)
    code = re.sub(r"[.\s\-]", "", _cell_str(v))
    if numeric and code.isdigit() and len(code) in (3, 5, 7):
        code = "0" + code
    return code


def find_codes_in_rows(rows):
    """Reads the HSN column below each item-table header in a sheet (list of rows).
    Stops at a Total row or the HSN tax-summary table. Returns items or None if the
    sheet has no HSN header at all."""
    items, found_header, i = [], False, 0
    n = len(rows)
    while i < n:
        cells = [_cell_str(c) for c in rows[i]]
        lower = [c.lower() for c in cells]
        hsn_col = next((k for k, c in enumerate(lower) if ("hsn" in c or "ritc" in c) and len(c) < 40), -1)
        if hsn_col == -1 or is_summary_header(" ".join(lower)):
            i += 1
            continue
        found_header = True
        sl_col = next((k for k, c in enumerate(lower) if SL_HEADER_RE.search(c)), -1)
        desc_col = _find_desc_col(lower, {hsn_col, sl_col})
        j = i + 1
        while j < n:
            row = rows[j]
            row_cells = [_cell_str(c) for c in row]
            joined = " ".join(row_cells).lower()
            first = next((c for c in row_cells if c), "").lower()
            if is_summary_header(joined) or re.match(r"^(sub\s*-?\s*|grand\s+)?total\b", first):
                break
            if any(("hsn" in c.lower() or "ritc" in c.lower()) and len(c) < 40 for c in row_cells):
                break  # another table header
            if len(row) > hsn_col:
                code = _excel_code(row[hsn_col])
                if code.isdigit() and len(code) in (4, 6, 8):
                    sl = ""
                    if sl_col != -1 and len(row) > sl_col:
                        m = re.match(r"\s*0*(\d{1,4})", _cell_str(row[sl_col]))
                        sl = m.group(1) if m else ""
                    desc = clean_desc(_cell_str(row[desc_col])) if desc_col != -1 and len(row) > desc_col else ""
                    items.append({"Sl. No.": sl, "Code": code, "Description": desc})
            j += 1
        i = j
    return items if found_header else None


def ensure_package(module, pip_name=None):
    """Import `module`, installing it into the Python that runs this app if missing
    (avoids 'Import openpyxl failed' when pip installed it into a different Python)."""
    try:
        return importlib.import_module(module)
    except ImportError:
        pass
    with st.spinner(f"Installing {pip_name or module} (one-time setup)..."):
        result = subprocess.run([sys.executable, "-m", "pip", "install", pip_name or module],
                                capture_output=True, text=True)
    importlib.invalidate_caches()
    try:
        return importlib.import_module(module)
    except ImportError:
        raise ImportError(
            f"Could not install '{pip_name or module}' automatically. Open Command Prompt and run:\n"
            f'"{sys.executable}" -m pip install {pip_name or module}\n\n{(result.stderr or "")[-500:]}'
        )


def process_spreadsheet(file_bytes, filename):
    """Returns (raw_text, items, notes)."""
    name = filename.lower()
    notes = []
    try:
        if name.endswith(".xls"):
            ensure_package("xlrd")
        elif not name.endswith(".csv"):
            ensure_package("openpyxl")
        if name.endswith(".csv"):
            raw = None
            for enc in ("utf-8-sig", "cp1252"):
                try:
                    raw = pd.read_csv(io.BytesIO(file_bytes), header=None, dtype=str, sep=None,
                                      engine="python", encoding=enc, keep_default_na=False)
                    break
                except UnicodeDecodeError:
                    continue
            sheets = {"CSV": raw}
        else:
            sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None, dtype=object)
    except ImportError as e:
        return "", [], [f"Missing package to read this file type. {e}"]
    except Exception as e:
        return "", [], [f"Could not read the spreadsheet — the file may be corrupted or not a real Excel/CSV file ({e})."]

    items, text_parts, used = [], [], []
    for sheet_name, df in sheets.items():
        if df is None or df.empty:
            continue
        rows = df.values.tolist()
        text_parts.append(f"----- Sheet: {sheet_name} -----")
        text_parts += ["\t".join(_cell_str(c) for c in r).rstrip() for r in rows]
        sheet_items = find_codes_in_rows(rows)
        if sheet_items is None:  # no HSN header: search the sheet like plain text
            sheet_items = find_codes_in_text("\n".join(" ".join(_cell_str(c) for c in r) for r in rows))
        if sheet_items:
            used.append(str(sheet_name))
            items += sheet_items
    if len(used) > 1:
        notes.append(f"ℹ️ HSN codes were found on {len(used)} sheets: {', '.join(used)}.")
    return "\n".join(text_parts), items, notes


def validate(items):
    rows = []
    next_sl = 1
    for i, item in enumerate(items):
        code = item["Code"]
        sl = item["Sl. No."]
        # Trust a Sl. No. read from the invoice only if it fits the running sequence
        # (OCR can misread 17 as 7 or 27 as 21); otherwise continue the count.
        if sl and sl.isdigit() and (i == 0 or next_sl <= int(sl) <= next_sl + 3):
            next_sl = int(sl) + 1
        else:
            sl = str(next_sl)
            next_sl += 1

        info = HSN_LOOKUP.get(code)
        if info and str(info.get("Status", "")).lower() == "outdated":
            status, desc, policy = "⚠️ Outdated (removed from official list)", info["Description"], info["Policy"]
        elif info:
            status, desc, policy = "✅ Valid", info["Description"], info["Policy"]
        elif len(code) < 8 and code in VALID_PREFIXES:
            status = f"⚠️ Incomplete ({len(code)} digits; DGFT needs 8)"
            desc, policy = "-", "-"
        else:
            status, desc, policy = "❌ Invalid / Not Found", "-", "-"

        row = {"Sl. No.": sl, "Item Description": item.get("Description", "") or "-",
               "HSN Code": code, "Status": status, "DGFT Description": desc}
        if HAS_POLICY_DATA:
            row["Policy"] = policy
        rows.append(row)
    return pd.DataFrame(rows)


def show_notes(notes):
    for note in notes:
        if note == "TESSERACT_MISSING":
            st.error(TESSERACT_MISSING_MSG)
        elif note.startswith("ℹ️"):
            st.caption(note)
        else:
            st.warning(note)


# ---------------------------------------------------------------------------
# HSN database updater (DGFT ITC(HS) schedule / CBIC Customs Tariff / Excel / CSV)
# ---------------------------------------------------------------------------
# The official sites have no public API (DGFT's ITC(HS) search is behind a CAPTCHA),
# so updates come from the official files: upload one, review the differences, apply.
SOURCE_DGFT, SOURCE_CBIC, SOURCE_OTHER = "DGFT ITC(HS) schedule", "CBIC Customs Tariff", "Other list (Excel / CSV)"
POLICY_RE = re.compile(r"\b(Free|Restricted|Prohibited|STE|State Trading Enterprises?)\b", re.I)
REF_CODE_LINE_RE = re.compile(r"^\s*(\d{4})\s?\.?\s?(\d{2})\s?\.?\s?(\d{2})(?!\d)\s*(.*)$")
_CBIC_TAIL_RE = re.compile(
    r"(\s+(u|kg\.?|g\.?|l\.?|m\.?|m2|m3|pa|1000\s*u|1000\s*kwh|tu|kg|carat|c/k|gi\s*f/s|ton(ne)?s?|"
    r"\d+(\.\d+)?%|free|-|–|\d+(\.\d+)?|rs\.?\s*\d+.*))+\s*$", re.I)


def _clean_ref_desc(text):
    text = re.sub(r"^[\s\-–:•]+", "", str(text or ""))
    return re.sub(r"\s+", " ", text).strip()[:300]


def parse_reference_file(file_bytes, filename, source):
    """Official file -> DataFrame[HSN Code, Description, Policy] (Policy '' if unknown)."""
    name = filename.lower()
    rows = []
    if name.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                for line in (page.extract_text() or "").split("\n"):
                    m = REF_CODE_LINE_RE.match(line)
                    if not m or m.group(1)[:2] == "00":
                        continue
                    code, rest = m.group(1) + m.group(2) + m.group(3), m.group(4)
                    policy = ""
                    if source == SOURCE_DGFT:
                        pm = POLICY_RE.search(rest)
                        if pm:
                            policy = pm.group(1).title().replace("Ste", "STE")
                            if policy.lower().startswith("state trading"):
                                policy = "STE"
                            rest = rest[: pm.start()]
                    else:
                        rest = _CBIC_TAIL_RE.sub("", rest)
                    rows.append((code, _clean_ref_desc(rest), policy))
    else:
        if name.endswith(".csv"):
            sheets = {"csv": pd.read_csv(io.BytesIO(file_bytes), header=None, dtype=object,
                                         sep=None, engine="python", encoding_errors="replace")}
        else:
            ensure_package("xlrd" if name.endswith(".xls") else "openpyxl")
            sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None, dtype=object)
        for df in sheets.values():
            data = df.values.tolist()
            code_col = desc_col = pol_col = -1
            start = 0
            for r_idx, row in enumerate(data[:30]):  # find a header row
                lower = [_cell_str(c).lower() for c in row]
                cc = next((k for k, c in enumerate(lower) if re.search(r"hsn|itc|tariff item|cth|hs code|code", c) and len(c) < 40), -1)
                if cc != -1:
                    code_col, start = cc, r_idx + 1
                    desc_col = next((k for k, c in enumerate(lower) if k != cc and re.search(r"desc|item|goods|commodity", c)), -1)
                    pol_col = next((k for k, c in enumerate(lower) if "policy" in c and "condition" not in c), -1)
                    break
            for row in data[start:]:
                if code_col == -1:  # no header: first cell with an 8-digit code, next text cell
                    for k, v in enumerate(row):
                        code = _excel_code(v)
                        if code.isdigit() and len(code) == 8:
                            text = next((_cell_str(x) for x in row[k + 1:] if re.search(r"[A-Za-z]", _cell_str(x))), "")
                            rows.append((code, _clean_ref_desc(text), ""))
                            break
                    continue
                if len(row) <= code_col:
                    continue
                code = _excel_code(row[code_col])
                if not (code.isdigit() and len(code) == 8):
                    continue
                desc = _cell_str(row[desc_col]) if desc_col != -1 and len(row) > desc_col else ""
                pol = _cell_str(row[pol_col]) if pol_col != -1 and len(row) > pol_col else ""
                pm = POLICY_RE.search(pol)
                rows.append((code, _clean_ref_desc(desc), pm.group(1).title() if pm else pol.strip()))
    ref = pd.DataFrame(rows, columns=["HSN Code", "Description", "Policy"])
    return ref.drop_duplicates(subset="HSN Code", keep="first").reset_index(drop=True)


def diff_reference(ref, db, full_list=False):
    """Compare an official list with the current database.
    Removals are only considered inside the chapters the file covers (a single chapter
    file must not mark the other 97 chapters outdated), unless `full_list` is set."""
    cur = db.set_index("HSN Code")
    ref_codes = set(ref["HSN Code"])
    chapters = {c[:2] for c in ref_codes}
    added, changed, outdated, reactivated = [], [], [], []
    for _, r in ref.iterrows():
        code = r["HSN Code"]
        if code not in cur.index:
            added.append(r.to_dict())
            continue
        old = cur.loc[code]
        if str(old.get("Status", "Active")).lower() == "outdated":
            reactivated.append({"HSN Code": code, "Description": r["Description"] or old["Description"]})
        new_desc = r["Description"] or old["Description"]
        new_pol = r["Policy"] or old["Policy"]
        if new_desc.strip().lower() != str(old["Description"]).strip().lower() or new_pol != old["Policy"]:
            changed.append({"HSN Code": code, "Old Description": old["Description"], "New Description": new_desc,
                            "Old Policy": old["Policy"], "New Policy": new_pol})
    # A chapter counts as fully covered only if the file lists at least half of the codes
    # we already have in it; otherwise it is a partial list and nothing there is outdated.
    active = cur[cur["Status"].str.lower() != "outdated"] if "Status" in cur else cur
    covered = set()
    for ch in chapters:
        in_db = {c for c in active.index if c[:2] == ch}
        if not in_db or len(in_db & ref_codes) / len(in_db) >= 0.5:
            covered.add(ch)
    for code, old in cur.iterrows():
        if code in ref_codes or str(old.get("Status", "Active")).lower() == "outdated":
            continue
        if full_list or code[:2] in covered:
            outdated.append({"HSN Code": code, "Description": old["Description"]})
    return dict(added=pd.DataFrame(added, columns=["HSN Code", "Description", "Policy"]),
                changed=pd.DataFrame(changed, columns=["HSN Code", "Old Description", "New Description", "Old Policy", "New Policy"]),
                outdated=pd.DataFrame(outdated, columns=["HSN Code", "Description"]),
                reactivated=pd.DataFrame(reactivated, columns=["HSN Code", "Description"]),
                chapters=sorted(chapters), partial=sorted(chapters - covered) if not full_list else [])


def apply_changes(db, diff, source_label, do_add=True, do_change=True, do_outdate=True):
    """Write the new hsn_master.csv (after a timestamped backup) and log every change."""
    import datetime as _dt
    now = _dt.datetime.now()
    today = now.strftime("%Y-%m-%d")
    BACKUP_DIR.mkdir(exist_ok=True)
    backup = BACKUP_DIR / f"hsn_master_{now.strftime('%Y%m%d_%H%M%S')}.csv"
    db.to_csv(backup, index=False)

    out = db.set_index("HSN Code").copy()
    log = []
    if do_add:
        for _, r in diff["added"].iterrows():
            out.loc[r["HSN Code"]] = {"Description": r["Description"], "Policy": r["Policy"] or "Unknown",
                                      "Status": "Active", "Updated": today}
            log.append((today, r["HSN Code"], "Added", "", r["Description"], source_label))
    if do_change:
        for _, r in diff["changed"].iterrows():
            out.loc[r["HSN Code"], ["Description", "Policy", "Updated"]] = [r["New Description"], r["New Policy"] or "Unknown", today]
            log.append((today, r["HSN Code"], "Updated", f'{r["Old Description"]} | {r["Old Policy"]}',
                        f'{r["New Description"]} | {r["New Policy"]}', source_label))
        for _, r in diff["reactivated"].iterrows():
            out.loc[r["HSN Code"], ["Status", "Updated"]] = ["Active", today]
            log.append((today, r["HSN Code"], "Re-activated", "Outdated", "Active", source_label))
    if do_outdate:
        for _, r in diff["outdated"].iterrows():
            out.loc[r["HSN Code"], ["Status", "Updated"]] = ["Outdated", today]
            log.append((today, r["HSN Code"], "Marked outdated", "Active", "Outdated", source_label))

    out = out.reset_index().rename(columns={"index": "HSN Code"}).sort_values("HSN Code")
    out[["HSN Code", "Description", "Policy", "Status", "Updated"]].to_csv(HSN_CSV, index=False)
    if log:
        pd.DataFrame(log, columns=["Date", "HSN Code", "Change", "Old", "New", "Source"]).to_csv(
            CHANGELOG_CSV, mode="a", header=not CHANGELOG_CSV.exists(), index=False)
    return len(log), backup


def _admin_unlocked():
    """Optional password (env var / Streamlit secret HSN_ADMIN_PASSWORD) for the updater."""
    pwd = os.environ.get("HSN_ADMIN_PASSWORD", "")
    try:
        pwd = pwd or st.secrets.get("HSN_ADMIN_PASSWORD", "")
    except Exception:
        pass
    if not pwd:
        return True
    if st.session_state.get("admin_ok"):
        return True
    entered = st.text_input("Admin password", type="password")
    if entered and entered == pwd:
        st.session_state["admin_ok"] = True
        return True
    if entered:
        st.error("Wrong password")
    return False


def render_updater():
    n_active = int((hsn_db["Status"] != "Outdated").sum())
    n_old = len(hsn_db) - n_active
    st.caption(f"{n_active:,} active codes" + (f" · {n_old:,} outdated" if n_old else ""))
    if not _admin_unlocked():
        return

    st.markdown("**Update from an official file**")
    source = st.selectbox("Source", [SOURCE_DGFT, SOURCE_CBIC, SOURCE_OTHER])
    st.caption(
        "DGFT: ITC(HS) Schedule-I/II files from dgft.gov.in (Regulatory Updates → ITC(HS)). "
        "CBIC: Customs Tariff chapter files from cbic.gov.in. PDF (text, not scanned), Excel or CSV."
    )
    ref_file = st.file_uploader("Official file", type=["pdf", "xlsx", "xls", "csv"], key="ref_file")
    full_list = st.checkbox("This file is the COMPLETE list (all chapters)", value=False,
                            help="Only then are codes missing from the file in other chapters marked outdated.")
    if ref_file is not None:
        with st.spinner("Reading the official file..."):
            try:
                ref = parse_reference_file(ref_file.getvalue(), ref_file.name, source)
            except Exception as e:
                st.error(f"Could not read the file: {e}")
                return
        if ref.empty:
            st.warning("No 8-digit HSN codes found in this file. If it is a scanned PDF, use the Excel/CSV version.")
            return
        d = diff_reference(ref, hsn_db, full_list)
        st.write(f"Read **{len(ref):,}** codes covering chapter(s) {', '.join(d['chapters'][:12])}"
                 + ("…" if len(d["chapters"]) > 12 else ""))
        if d["partial"]:
            st.caption(f"Chapter(s) {', '.join(d['partial'])} are only partly in this file, so no codes "
                       "there are marked outdated.")
        c1, c2, c3 = st.columns(3)
        c1.metric("New", len(d["added"]))
        c2.metric("Changed", len(d["changed"]) + len(d["reactivated"]))
        c3.metric("Outdated", len(d["outdated"]))
        if len(d["added"]):
            with st.expander(f"New codes ({len(d['added'])})"):
                st.dataframe(d["added"], hide_index=True, use_container_width=True)
        if len(d["changed"]) or len(d["reactivated"]):
            with st.expander(f"Changed codes ({len(d['changed']) + len(d['reactivated'])})"):
                st.dataframe(d["changed"], hide_index=True, use_container_width=True)
                if len(d["reactivated"]):
                    st.caption("Back in the official list (will be re-activated):")
                    st.dataframe(d["reactivated"], hide_index=True, use_container_width=True)
        if len(d["outdated"]):
            with st.expander(f"Outdated codes ({len(d['outdated'])}) — not in this file"):
                st.dataframe(d["outdated"], hide_index=True, use_container_width=True)
        do_add = st.checkbox("Add new codes", value=True)
        do_change = st.checkbox("Update changed descriptions / policy", value=True)
        do_outdate = st.checkbox("Mark missing codes as outdated", value=True)
        total = (len(d["added"]) * do_add + (len(d["changed"]) + len(d["reactivated"])) * do_change
                 + len(d["outdated"]) * do_outdate)
        if st.button(f"Apply {total} change(s)", type="primary", disabled=total == 0):
            n, backup = apply_changes(hsn_db, d, f"{source}: {ref_file.name}", do_add, do_change, do_outdate)
            load_master_hsn_db.clear()
            st.session_state["update_msg"] = f"✅ Applied {n} change(s). Backup saved as {backup.name}."
            st.rerun()

    st.divider()
    st.markdown("**Add or edit one code**")
    with st.form("manual_code", clear_on_submit=True):
        code = st.text_input("HSN code (8 digits)", max_chars=10)
        desc = st.text_input("Description")
        pol = st.selectbox("Policy", ["Unknown", "Free", "Restricted", "Prohibited", "STE"])
        status = st.selectbox("Status", ["Active", "Outdated"])
        if st.form_submit_button("Save code"):
            code = re.sub(r"\D", "", code)
            if len(code) != 8:
                st.error("Enter an 8-digit code.")
            else:
                existing = HSN_LOOKUP.get(code)
                d = dict(added=pd.DataFrame(columns=["HSN Code", "Description", "Policy"]),
                         changed=pd.DataFrame(columns=["HSN Code", "Old Description", "New Description", "Old Policy", "New Policy"]),
                         outdated=pd.DataFrame(columns=["HSN Code", "Description"]),
                         reactivated=pd.DataFrame(columns=["HSN Code", "Description"]))
                if existing is None:
                    d["added"] = pd.DataFrame([{"HSN Code": code, "Description": desc, "Policy": pol}])
                else:
                    d["changed"] = pd.DataFrame([{"HSN Code": code, "Old Description": existing["Description"],
                                                  "New Description": desc or existing["Description"],
                                                  "Old Policy": existing["Policy"], "New Policy": pol}])
                if status == "Outdated":
                    d["outdated"] = pd.DataFrame([{"HSN Code": code, "Description": desc}])
                elif existing is not None and str(existing.get("Status")).lower() == "outdated":
                    d["reactivated"] = pd.DataFrame([{"HSN Code": code, "Description": desc}])
                db_for_apply = hsn_db
                if existing is None and status == "Outdated":  # add first, then mark
                    apply_changes(hsn_db, {**d, "outdated": d["outdated"].iloc[0:0]}, "Manual entry")
                    load_master_hsn_db.clear()
                    db_for_apply, _ = load_master_hsn_db()
                    d = {**d, "added": d["added"].iloc[0:0]}
                apply_changes(db_for_apply, d, "Manual entry")
                load_master_hsn_db.clear()
                st.session_state["update_msg"] = f"✅ Saved {code}."
                st.rerun()

    if CHANGELOG_CSV.exists():
        with st.expander("Change history"):
            log = pd.read_csv(CHANGELOG_CSV, dtype=str).fillna("")
            st.dataframe(log.iloc[::-1].head(500), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def show_results(extracted_text, items, notes):
    show_notes(notes)
    if not items:
        if not extracted_text.strip():
            st.warning("Could not read any text from this document.")
        else:
            st.warning("No HSN codes found in this document.")
        return

    results_df = validate(items)
    n_total = len(results_df)
    n_valid = int(results_df["Status"].str.startswith("✅").sum())
    n_warn = int(results_df["Status"].str.startswith("⚠️").sum())
    n_bad = n_total - n_valid - n_warn

    c1, c2, c3 = st.columns(3)
    c1.metric("Items", n_total)
    c2.metric("Valid", n_valid)
    c3.metric("Need attention", n_bad + n_warn)

    def highlight_rows(row):
        if row["Status"].startswith("✅"):
            color = "rgba(0, 200, 0, 0.08)"
        elif row["Status"].startswith("⚠️"):
            color = "rgba(255, 200, 0, 0.15)"
        else:
            color = "rgba(255, 0, 0, 0.10)"
        return [f"background-color: {color}"] * len(row)

    st.dataframe(
        results_df.style.apply(highlight_rows, axis=1),
        hide_index=True,
        use_container_width=True,
        height=min(38 * (n_total + 1) + 3, 700),
    )


def process_upload(name, file_bytes, mime=""):
    lower = name.lower()
    with st.spinner("Checking HSN codes..."):
        try:
            if lower.endswith(".pdf") or mime == "application/pdf":
                text, items, notes, all_ocred = process_pdf(file_bytes)
                # Last resort: corrupted / glued text layer -> OCR every page
                # (skipped when every page was already OCR'd, e.g. a fully scanned PDF)
                if not items and not all_ocred and tesseract_available():
                    text, items, ocr_notes, _ = process_pdf(file_bytes, force_ocr=True)
                    notes += [n for n in ocr_notes if n not in notes]
            elif lower.endswith((".xlsx", ".xlsm", ".xls", ".csv")):
                text, items, notes = process_spreadsheet(file_bytes, name)
            else:
                text, items, notes = process_image(file_bytes)
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            return
    show_results(text, items, notes)


with st.sidebar:
    st.subheader("🔄 HSN database")
    if st.session_state.get("update_msg"):
        st.success(st.session_state.pop("update_msg"))
    render_updater()

# A radio (not st.tabs) so the camera only starts when "Take photo" is chosen:
# st.tabs renders every tab at once, which switched the webcam on immediately.
mode = st.radio("Input", ["📄 Upload invoice", "📷 Take photo"], horizontal=True, label_visibility="collapsed")

if mode.endswith("Upload invoice"):
    uploaded_file = st.file_uploader(
        "PDF, Excel, CSV or image",
        type=["pdf", "xlsx", "xlsm", "xls", "csv", "png", "jpg", "jpeg"],
    )
    if uploaded_file is not None:
        process_upload(uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
else:
    photo = st.camera_input("Fill the frame with the item table, in even light")
    if photo is not None:
        process_upload("camera_photo.jpg", photo.getvalue(), "image/jpeg")

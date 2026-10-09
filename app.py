import io
import os
import re
import shutil
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

st.set_page_config(page_title="HSN/RITC Validator", layout="wide")
st.title("DGFT HSN/RITC Code Validator")
st.write("Upload an invoice (PDF, image or Excel) or take a photo of it to extract and validate HSN/RITC codes against the DGFT database.")


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
    df["HSN Code"] = df["HSN Code"].str.strip()
    df = df[df["HSN Code"] != ""].drop_duplicates(subset="HSN Code")
    return df, None


hsn_db, db_error = load_master_hsn_db()
if db_error:
    st.error(f"❌ HSN Database could not be loaded. {db_error}")
    st.stop()

HSN_LOOKUP = hsn_db.set_index("HSN Code")[["Description", "Policy"]].to_dict("index")
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


def _clean_crop(crop, pad=30):
    """Erase rule lines and add a white margin: Tesseract tends to stop reading a
    column part-way down when row lines touch the text."""
    a = np.asarray(crop, dtype=np.uint8).copy()
    if a.size:
        a[(a < 128).mean(axis=1) > 0.5, :] = 255
        a[:, (a < 128).mean(axis=0) > 0.5] = 255
    return ImageOps.expand(Image.fromarray(a), border=pad, fill=255), pad


def ocr_page_text(gray):
    """One --psm 3 pass. Returns (text, words, search_text).
    `search_text` drops everything physically below the HSN tax-summary header, so the
    summary is never counted even when Tesseract outputs its block before the items."""
    words = _ocr_words(gray, "--psm 3")
    lines, order = {}, []
    for w in words:
        if w["line"] not in lines:
            order.append(w["line"])
        lines.setdefault(w["line"], []).append(w)
    line_text = {k: " ".join(w["text"] for w in lines[k]) for k in order}
    line_top = {k: min(w["y"] for w in lines[k]) for k in order}
    summary_tops = [line_top[k] for k in order if is_summary_header(line_text[k])]
    cutoff = min(summary_tops) if summary_tops else None

    def build(keep):
        out, prev_block = [], None
        for k in order:
            if not keep(k):
                continue
            if prev_block is not None and k[0] != prev_block:
                out.append("")
            out.append(line_text[k])
            prev_block = k[0]
        return "\n".join(out)

    text = build(lambda k: True)
    search_text = build(lambda k: cutoff is None or line_top[k] < cutoff - 5)
    return text, words, search_text


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
        items.append({"Sl. No.": sl, "Code": code})
    return items


def ocr_page_items(gray, text, words, search_text=None):
    """Find HSN codes on an OCR'd page, trying several readings and keeping the one with
    the most valid codes:
    - the --psm 3 text (automatic layout): most scans
    - column OCR: dense ruled tables where full-page OCR drops or misreads rows
    - --psm 6 (one uniform block): grid-heavy scans where psm 3 skips whole columns
    Returns (text, items)."""
    best = find_codes_in_text(search_text if search_text is not None else text)
    try:
        col_items = ocr_table_columns(gray, words)
    except Exception:
        col_items = []
    if col_items and _count_valid(col_items) >= _count_valid(best):
        best = col_items

    if _count_valid(best) == 0:
        text6 = pytesseract.image_to_string(gray, config="--psm 6")
        items6 = find_codes_in_text(text6)
        if _count_valid(items6) > 0:
            return text6, items6
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
            line_hits.append({"Sl. No.": sl, "Code": code})

        # An item row has exactly one HSN cell and it comes after the description, so if
        # a line has several candidates (e.g. "CURTAIN - 63031100 ... 63039990 187 PCS")
        # only the last one is the real HSN code.
        if line_hits:
            results.append(line_hits[-1])
    return results


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
        hsn_col = sl_col = header_row = -1
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
                break

        if hsn_col != -1:
            if is_summary:
                continue  # skip the HSN tax-summary table
            state.update(hsn_col=hsn_col, sl_col=sl_col, ncols=len(table[header_row]))
        elif state.get("hsn_col", -1) != -1 and len(table[0] or []) == state["ncols"]:
            # Continuation table on a later page with the same column layout
            hsn_col, sl_col, header_row = state["hsn_col"], state["sl_col"], -1
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
            results.append({"Sl. No.": sl, "Code": code})
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
                    items.append({"Sl. No.": sl, "Code": code})
            j += 1
        i = j
    return items if found_header else None


def process_spreadsheet(file_bytes, filename):
    """Returns (raw_text, items, notes)."""
    name = filename.lower()
    notes = []
    try:
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
        return "", [], [f"Missing package to read this file type: {e}. Run: pip install -r requirements.txt"]
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
        if info:
            status, desc, policy = "✅ Valid", info["Description"], info["Policy"]
        elif len(code) < 8 and code in VALID_PREFIXES:
            status = f"⚠️ Incomplete ({len(code)} digits; DGFT needs 8)"
            desc, policy = "-", "-"
        else:
            status, desc, policy = "❌ Invalid / Not Found", "-", "-"

        row = {"Sl. No.": sl, "Extracted Code": code, "Status": status, "DGFT Description": desc}
        if HAS_POLICY_DATA:
            row["Policy"] = policy
        rows.append(row)
    return pd.DataFrame(rows)


def show_notes(notes):
    for note in notes:
        if note == "TESSERACT_MISSING":
            st.error(TESSERACT_MISSING_MSG)
        else:
            st.warning(note)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def show_results(source_name, extracted_text, items, notes):
    show_notes(notes)
    if not extracted_text.strip() and not items:
        st.warning("Could not extract any text from the document.")
        return
    with st.expander("View Raw Extracted Text"):
        st.text(extracted_text)
    if not items:
        st.warning("No potential HSN codes found in the document.")
        return

    results_df = validate(items)
    n_valid = results_df["Status"].str.startswith("✅").sum()
    st.subheader(f"Found {len(results_df)} HSN codes — {n_valid} valid, {len(results_df) - n_valid} need attention")

    def highlight_rows(row):
        if row["Status"].startswith("✅"):
            color = "rgba(0, 255, 0, 0.1)"
        elif row["Status"].startswith("⚠️"):
            color = "rgba(255, 200, 0, 0.15)"
        else:
            color = "rgba(255, 0, 0, 0.1)"
        return [f"background-color: {color}"] * len(row)

    st.dataframe(results_df.style.apply(highlight_rows, axis=1), hide_index=True, use_container_width=True)

    stem = Path(source_name).stem
    c1, c2 = st.columns(2)
    with c1:
        xlsx = io.BytesIO()
        try:
            with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
                results_df.to_excel(writer, index=False, sheet_name="HSN Check")
                ws = writer.sheets["HSN Check"]
                for col, width in zip("ABCDE", (8, 16, 38, 70, 14)):
                    ws.column_dimensions[col].width = width
            st.download_button("Download results (Excel)", xlsx.getvalue(), file_name=f"{stem}_hsn_check.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key=f"xlsx_{source_name}")
        except ImportError:
            st.caption("Install openpyxl for Excel download: pip install openpyxl")
    with c2:
        st.download_button("Download results (CSV)", results_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{stem}_hsn_check.csv", mime="text/csv", key=f"csv_{source_name}")


def process_upload(name, file_bytes, mime=""):
    lower = name.lower()
    with st.spinner("Processing document..."):
        try:
            if lower.endswith(".pdf") or mime == "application/pdf":
                text, items, notes, all_ocred = process_pdf(file_bytes)
                # Last resort: corrupted / glued text layer -> OCR every page
                # (skipped when every page was already OCR'd, e.g. a fully scanned PDF)
                if not items and not all_ocred and tesseract_available():
                    st.info("Native text extraction found no codes (possibly glued columns or a garbled text layer). Retrying with full-page OCR...")
                    text, items, ocr_notes, _ = process_pdf(file_bytes, force_ocr=True)
                    notes += [n for n in ocr_notes if n not in notes]
            elif lower.endswith((".xlsx", ".xlsm", ".xls", ".csv")):
                text, items, notes = process_spreadsheet(file_bytes, name)
            else:
                text, items, notes = process_image(file_bytes)
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            return
    show_results(name, text, items, notes)


tab_upload, tab_photo = st.tabs(["📄 Upload invoice", "📷 Take photo"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload Invoice (PDF, Excel, CSV, PNG, JPG)",
        type=["pdf", "xlsx", "xlsm", "xls", "csv", "png", "jpg", "jpeg"],
    )
    if uploaded_file is not None:
        process_upload(uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)

with tab_photo:
    st.caption(
        "Hold the invoice flat, fill the frame with the item table, and make sure the light is even "
        "(no shadows or glare). Webcams are low resolution — for small print, take the photo with "
        "your phone camera and upload it in the **Upload invoice** tab instead."
    )
    photo = st.camera_input("Take a photo of the invoice")
    if photo is not None:
        process_upload("camera_photo.jpg", photo.getvalue(), "image/jpeg")

st.markdown("---")
note = f"**Note:** Codes are checked against `{HSN_CSV.name}` ({len(HSN_LOOKUP):,} ITC(HS) codes)."
if not HAS_POLICY_DATA:
    note += " The database has no import/export policy data (all 'Unknown'), so only code validity is checked."
note += " Scanned PDFs, images and photos require Tesseract OCR on this machine."
st.markdown(note)

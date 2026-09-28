"""
kb.py — Knowledge-base layer for MedSupply Sentinel (no Streamlit code here).

Ported from notebook section 3B ("upload, merge/update and use as reference").

HOW IT REACHES THE MAIN PAGE (app.py) WITHOUT TOUCHING app.py / sentinel_core.py
--------------------------------------------------------------------------------
sentinel_core reads its data from CSV files (synthetic/*.csv) and PDFs (guidelines/*.pdf)
at call time. This module upserts uploaded rows INTO THOSE SAME FILES, so the main
workflow picks the new data up on its next run automatically.

  * Tables  : existing key -> updated, new key -> added (blank cell = leave unchanged).
  * PDFs    : copied into guidelines/ with the prefix "uploaded_".
  * TXT/MD  : converted to a small PDF in guidelines/ (sentinel_core only reads PDFs).
  * Unknown CSV/Excel: converted to searchable reference text (PDF) as well.
  * Baseline: the original CSVs are backed up once, so reset() restores them.
  * shortage_events.csv is written "latest report first per drug", because sentinel_core's
    LocalShortageAdapter takes the FIRST row of a drug (so the newest event wins).
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape

import pandas as pd
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

import sentinel_core as core

# ------------------------------------------------------------------ paths
KB_DIR = core.BASE_DIR / "knowledge_base"
BASELINE_DIR = KB_DIR / "baseline"       # original CSVs (for reset)
UPLOAD_DIR = KB_DIR / "uploads"          # raw copy of every uploaded file (audit trail)
MANIFEST = KB_DIR / "manifest.json"      # ingest history
UPLOADED_PREFIX = "uploaded_"            # prefix of every document this module adds to guidelines/
for _d in (KB_DIR, BASELINE_DIR, UPLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

_LOCK = threading.RLock()

# Primary keys used for "update if exists, otherwise add"
TABLE_KEYS: Dict[str, List[str]] = {
    "patients": ["patient_id"],
    "medications": ["drug_id"],
    "inventory": ["drug_id"],
    "suppliers": ["supplier_id", "medication"],
    "shortage_events": ["drug_id", "reported_date"],
}
_UPPER_COLUMNS = {"shortage_status", "supplier_status", "risk_group"}
_ALLOWED_SHORTAGE = {"NORMAL", "SHORTAGE", "CRITICAL_SHORTAGE"}
DOC_EXTENSIONS = {".pdf", ".txt", ".md"}
TABLE_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
ACCEPTED_TYPES = [e.lstrip(".") for e in sorted(DOC_EXTENSIONS | TABLE_EXTENSIONS)]


def _table_path(name: str) -> Path:
    return Path(core._TABLE_SPECS[name]["path"])


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)[:100] or "file"


# ------------------------------------------------------------------ reading helpers
def read_tables() -> Dict[str, pd.DataFrame]:
    """Current (merged) tables, exactly as sentinel_core sees them."""
    return {n: core.load_local_table(n) for n in core._TABLE_SPECS}


@lru_cache(maxsize=256)
def _pdf_pages_cached(path_str: str, mtime: float):
    reader = PdfReader(path_str)
    return tuple((i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1))


def read_pdf_pages(path) -> List[tuple]:
    p = Path(path)
    return list(_pdf_pages_cached(str(p), p.stat().st_mtime))


def guideline_files() -> List[tuple]:
    """[(origin, path)] for every guideline PDF; origin = 'dataset' or 'uploaded'."""
    d = Path(core.GUIDELINE_DIR)
    if not d.is_dir():
        return []
    return [("uploaded" if f.name.startswith(UPLOADED_PREFIX) else "dataset", f) for f in sorted(d.glob("*.pdf"))]


def data_fingerprint() -> tuple:
    """Changes whenever a table CSV or a guideline PDF changes (used to rebuild the chat index)."""
    fp = []
    paths = [_table_path(n) for n in core._TABLE_SPECS] + [f for _, f in guideline_files()]
    for p in paths:
        try:
            st = p.stat()
            fp.append((p.name, st.st_mtime_ns, st.st_size))
        except OSError:
            fp.append((p.name, 0, 0))
    return tuple(fp)


# ------------------------------------------------------------------ baseline / manifest / db sync
def ensure_baseline() -> None:
    """Back up the original CSVs once (before the first upload ever touches them)."""
    for name in core._TABLE_SPECS:
        src, dst = _table_path(name), BASELINE_DIR / f"{name}.csv"
        if not dst.is_file() and src.is_file():
            shutil.copy2(src, dst)


def _baseline_counts() -> Dict[str, int]:
    out = {}
    for name in core._TABLE_SPECS:
        b = BASELINE_DIR / f"{name}.csv"
        try:
            out[name] = len(core.load_local_table(name, b)) if b.is_file() else None
        except Exception:
            out[name] = None
    return out


def history() -> list:
    if MANIFEST.is_file():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_history(items: list) -> None:
    MANIFEST.write_text(json.dumps(items[-200:], indent=2, ensure_ascii=False), encoding="utf-8")


def sync_db() -> None:
    """Mirror the merged CSVs into the SQLite reference tables (best effort)."""
    try:
        core.init_db()
    except Exception as exc:
        print(f"[KB] SQLite sync skipped: {exc}")


# ------------------------------------------------------------------ row validation + upsert
def _validate_row(table: str, row: dict) -> dict:
    """Coerce the provided cells of one uploaded row. Blank cell = 'leave unchanged'."""
    spec, keys = core._TABLE_SPECS[table], TABLE_KEYS[table]
    out: Dict[str, Any] = {}
    for col, val in row.items():
        if col not in spec["columns"]:
            continue
        if pd.isna(val) or str(val).strip() == "":
            if col in keys:
                raise ValueError(f"empty key column '{col}'")
            continue
        s = str(val).strip()
        if col == "active_status":
            if s.lower() not in core._ACTIVE_STATUS_MAP:
                raise ValueError(f"unsupported active_status {val!r}")
            out[col] = core._ACTIVE_STATUS_MAP[s.lower()]
        elif col in spec["int"]:
            try:
                f = float(s)
            except ValueError:
                raise ValueError(f"'{col}' must be a whole number, got {val!r}")
            if f % 1 != 0 or f < 0:
                raise ValueError(f"'{col}' must be a non-negative whole number, got {val!r}")
            if col == "severity" and f > 100:
                raise ValueError("'severity' must be between 0 and 100")
            out[col] = int(f)
        elif col in spec["float"]:
            try:
                f = float(s)
            except ValueError:
                raise ValueError(f"'{col}' must be a number, got {val!r}")
            if f < 0:
                raise ValueError(f"'{col}' must be >= 0")
            out[col] = f
        else:
            if col in _UPPER_COLUMNS:
                s = re.sub(r"\s+", "_", s.upper())
            if col == "shortage_status" and s not in _ALLOWED_SHORTAGE:
                raise ValueError(f"shortage_status must be one of {sorted(_ALLOWED_SHORTAGE)}, got {val!r}")
            out[col] = s
    if table == "patients" and "patient_id" in out and not out["patient_id"].startswith("SYN-"):
        raise ValueError("patient_id must start with 'SYN-' (synthetic patients only — real patient data is refused)")
    return out


def upsert_table(current: pd.DataFrame, table: str, incoming: pd.DataFrame):
    """Return (new_dataframe, stats). Existing key -> update provided cells; new key -> add full record."""
    spec, keys = core._TABLE_SPECS[table], TABLE_KEYS[table]
    all_cols = spec["columns"]
    records = current.to_dict("records")
    index = {tuple(r[k] for k in keys): i for i, r in enumerate(records)}
    added = updated = 0
    rejected: List[str] = []
    ignored = [c for c in incoming.columns if c not in all_cols]

    for n, row in enumerate(incoming.to_dict("records"), start=2):      # row 1 = header
        try:
            clean = _validate_row(table, row)
            missing_keys = [k for k in keys if k not in clean]
            if missing_keys:
                raise ValueError(f"missing key value(s) {missing_keys}")
        except ValueError as e:
            rejected.append(f"row {n}: {e}")
            continue
        key = tuple(clean[k] for k in keys)
        if table == "medications":
            drug_name = clean.get("drug_name")
            if drug_name is None and key in index:
                drug_name = records[index[key]].get("drug_name")
            normalized_name = str(drug_name or "").strip().casefold()
            duplicate_name = any(
                i != index.get(key, -1)
                and str(record.get("drug_name", "")).strip().casefold() == normalized_name
                for i, record in enumerate(records)
            )
            if normalized_name and duplicate_name:
                rejected.append(f"row {n}: drug_name '{drug_name}' already belongs to another drug_id")
                continue
        if key in index:                                    # UPDATE only the provided cells
            records[index[key]].update(clean)
            updated += 1
        else:                                               # ADD needs a complete record
            missing = [c for c in all_cols if c not in clean]
            if missing:
                rejected.append(f"row {n}: new record {key} is missing columns {missing}")
                continue
            records.append({c: clean[c] for c in all_cols})
            index[key] = len(records) - 1
            added += 1

    new_df = pd.DataFrame(records, columns=all_cols)
    for c in spec["int"]:
        new_df[c] = new_df[c].astype("int64")
    for c in spec["float"]:
        new_df[c] = new_df[c].astype("float64")
    return new_df.reset_index(drop=True), {"added": added, "updated": updated,
                                           "rejected": rejected, "ignored_columns": ignored}


def _order_for_core(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """sentinel_core takes the FIRST shortage row of a drug -> put the newest report first per drug."""
    if table != "shortage_events" or df.empty:
        return df
    d = df.copy()
    d["_o"] = pd.factorize(d["drug_id"])[0]
    d["_d"] = pd.to_datetime(d["reported_date"], errors="coerce")
    d = d.sort_values(["_o", "_d"], ascending=[True, False], na_position="last", kind="stable")
    return d.drop(columns=["_o", "_d"]).reset_index(drop=True)


def _write_table(table: str, df: pd.DataFrame) -> None:
    _order_for_core(table, df).to_csv(_table_path(table), index=False)


def detect_table(columns, filename: str = ""):
    """Return (table_name, None) or (None, reason). Reason only when the file 'almost' matches."""
    cols, fname = set(columns), filename.lower()
    best = None
    for name, spec in core._TABLE_SPECS.items():
        overlap = len(cols & set(spec["columns"]))
        score = overlap + (0.5 if name.split("_")[0] in fname else 0.0)
        if best is None or score > best[0]:
            best = (score, name, overlap)
    _, name, overlap = best
    keys = TABLE_KEYS[name]
    if overlap < 2:
        return None, None
    missing_keys = [k for k in keys if k not in cols]
    if missing_keys:
        return None, f"looks like a '{name}' file but key column(s) {missing_keys} are missing"
    if overlap <= len(keys):
        return None, None
    return name, None


# ------------------------------------------------------------------ documents
def _text_to_pdf(text: str, dest: Path, title: str) -> None:
    """sentinel_core only reads PDFs, so TXT/MD/unknown tables become a small text PDF."""
    body = text.strip()
    safe = body.encode("latin-1", "replace").decode("latin-1")       # built-in PDF fonts are Latin-1 only
    lost = sum(1 for a, b in zip(body, safe) if a != b)
    if not body:
        raise ValueError("the file is empty")
    if lost > 0.3 * len(body):
        raise ValueError("mostly non-Latin text (e.g. Arabic) cannot be embedded in the guideline PDF; "
                         "upload it as a PDF instead")
    styles = getSampleStyleSheet()
    story = [Paragraph(escape(title), styles["Heading2"]), Spacer(1, 8)]
    for para in re.split(r"\n\s*\n", safe):
        para = para.strip()
        if para:
            story += [Paragraph(escape(para).replace("\n", "<br/>"), styles["BodyText"]), Spacer(1, 6)]
    SimpleDocTemplate(str(dest), pagesize=A4).build(story)


def _table_to_text(df: pd.DataFrame, title: str) -> str:
    rows = [f"Reference table: {title}"]
    for _, r in df.iterrows():
        cells = [f"{c}: {v}" for c, v in r.items() if str(v).strip() not in ("", "nan", "None")]
        if cells:
            rows.append(" | ".join(cells))
    return "\n\n".join(rows)


def _read_table_file(path: Path):
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        return list(pd.read_excel(path, sheet_name=None, dtype=str).items())
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return [(None, pd.read_csv(path, sep=None, engine="python", dtype=str, encoding=enc))]
        except UnicodeDecodeError:
            continue
    raise ValueError("cannot decode the file (try saving it as UTF-8 CSV)")


# ------------------------------------------------------------------ ingestion (one file at a time)
def ingest_file(path, original_name: Optional[str] = None) -> dict:
    """Add ONE uploaded file. Returns a report dict (never raises)."""
    p = Path(path)
    name = original_name or p.name
    ext = Path(name).suffix.lower()
    report: Dict[str, Any] = {"file": name, "kind": "", "details": [], "rejected": []}
    with _LOCK:
        try:
            if not p.is_file():
                raise FileNotFoundError(f"file not found: {p}")
            ensure_baseline()
            stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy2(p, UPLOAD_DIR / f"{stamp}_{_safe_name(name)}")            # audit copy
            Path(core.GUIDELINE_DIR).mkdir(parents=True, exist_ok=True)

            if ext in DOC_EXTENSIONS:
                stem = _safe_name(Path(name).stem)
                dest = Path(core.GUIDELINE_DIR) / f"{UPLOADED_PREFIX}{stem}.pdf"
                if ext == ".pdf":
                    reader = PdfReader(str(p))
                    chars = sum(len(pg.extract_text() or "") for pg in reader.pages)
                    if chars == 0:
                        raise ValueError("no extractable text (scanned PDF?)")
                    shutil.copy2(p, dest)
                    info = f"{len(reader.pages)} page(s), {chars} characters"
                else:
                    raw = p.read_text(encoding="utf-8", errors="ignore")
                    _text_to_pdf(raw, dest, Path(name).stem)
                    info = f"{len(raw)} characters (converted to PDF)"
                report.update(kind="document",
                              details=[f"added to guidelines as '{dest.name}': {info}"])

            elif ext in TABLE_EXTENSIONS:
                report["kind"] = "table"
                for sheet, df in _read_table_file(p):
                    label = f"{name}" + (f" [{sheet}]" if sheet else "")
                    df.columns = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in df.columns]
                    df = df.dropna(how="all")
                    if df.empty:
                        report["details"].append(f"{label}: empty, skipped")
                        continue
                    table, reason = detect_table(df.columns, name)
                    if table:
                        new_df, r = upsert_table(core.load_local_table(table), table, df)
                        if r["added"] or r["updated"]:
                            _write_table(table, new_df)
                        report["details"].append(
                            f"{label} -> table '{table}': {r['added']} added, {r['updated']} updated, "
                            f"{len(r['rejected'])} rejected"
                            + (f" (ignored columns: {r['ignored_columns']})" if r["ignored_columns"] else ""))
                        report["rejected"] += [f"{label}: {x}" for x in r["rejected"]]
                    elif reason:
                        report["rejected"].append(f"{label}: {reason} — nothing imported")
                    else:
                        suffix = f"_{_safe_name(sheet)}" if sheet else ""
                        dest = Path(core.GUIDELINE_DIR) / f"{UPLOADED_PREFIX}{_safe_name(Path(name).stem)}{suffix}_table.pdf"
                        _text_to_pdf(_table_to_text(df, label), dest, label)
                        report["details"].append(
                            f"{label}: columns not recognised as a known table -> stored as searchable "
                            f"reference document '{dest.name}' ({len(df)} rows)")
            else:
                raise ValueError(f"unsupported file type '{ext}' (use CSV, XLSX, PDF, TXT or MD)")
        except Exception as e:
            report["kind"] = "error"
            report["rejected"].append(f"{name}: {type(e).__name__}: {e}")

        report["time"] = core.now_iso()
        _save_history(history() + [report])
        sync_db()
    return report


def ingest_bytes(name: str, data: bytes) -> dict:
    """Ingest an uploaded file given as bytes (what Streamlit's file_uploader gives us)."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / _safe_name(name)
        p.write_bytes(data)
        return ingest_file(p, original_name=name)


def reset() -> None:
    """Throw away everything uploaded and restore the original dataset."""
    with _LOCK:
        for name in core._TABLE_SPECS:
            b = BASELINE_DIR / f"{name}.csv"
            if b.is_file():
                shutil.copy2(b, _table_path(name))
        gdir = Path(core.GUIDELINE_DIR)
        if gdir.is_dir():
            for f in gdir.glob(f"{UPLOADED_PREFIX}*"):
                f.unlink(missing_ok=True)
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        MANIFEST.unlink(missing_ok=True)
        sync_db()


# ------------------------------------------------------------------ status helpers
def status() -> pd.DataFrame:
    base = _baseline_counts()
    rows = []
    for n in core._TABLE_SPECS:
        now = len(core.load_local_table(n))
        orig = base.get(n)
        rows.append({"table": n, "rows_now": now,
                     "rows_in_original_dataset": orig if orig is not None else now,
                     "delta": now - (orig if orig is not None else now),
                     "key": " + ".join(TABLE_KEYS[n])})
    return pd.DataFrame(rows)


def documents() -> pd.DataFrame:
    rows = []
    for origin, f in guideline_files():
        try:
            parts = read_pdf_pages(f)
            rows.append({"document": f.name, "origin": origin, "pages": len(parts),
                         "characters": sum(len(t) for _, t in parts)})
        except Exception:
            rows.append({"document": f.name, "origin": origin, "pages": 0, "characters": 0})
    return pd.DataFrame(rows, columns=["document", "origin", "pages", "characters"])


def format_reports(reports) -> str:
    lines = []
    for r in reports:
        icon = "❌" if r["kind"] == "error" else ("⚠️" if r["rejected"] else "✅")
        lines.append(f"{icon} **{r['file']}**")
        lines += [f"  - {d}" for d in r["details"]]
        lines += [f"  - ⚠️ {x}" for x in r["rejected"][:8]]
        if len(r["rejected"]) > 8:
            lines.append(f"  - ... and {len(r['rejected']) - 8} more rejected rows")
    return "\n".join(lines) if lines else "Nothing to import."


UPLOAD_HELP = """
| File | What happens |
|---|---|
| `inventory.csv` (drug_id, drug_name, current_stock, daily_usage, reorder_level, supplier_id, unit_price, last_restock_date) | existing `drug_id` → **updated**, new `drug_id` → **added** |
| `suppliers.csv` (supplier_id, supplier_name, medication, available_quantity, unit_price, lead_time_days, supplier_status) | key = supplier_id + medication |
| `shortage_events.csv` (drug_id, drug_name, shortage_status, reported_date, source, severity, notes) | key = drug_id + reported_date; the **latest date** is the current status |
| `medications.csv` (drug_id, drug_name, category, demo_status) | key = drug_id |
| `synthetic_patients.csv` (patient_id, age_group, condition_category, medication, risk_group, active_status) | `patient_id` must start with `SYN-` (synthetic only) |
| `.pdf` / `.txt` / `.md` | added as guideline / reference documents |
| any other CSV / Excel | stored as searchable reference text |

Partial updates are fine: to update only the stock, upload `drug_id,current_stock`. Blank cells = *leave unchanged*.
A brand-new record needs all columns. A new drug needs **medications + inventory + shortage_events** (and suppliers
if you want a purchase order) before it can go through the main workflow.
"""

# ------------------------------------------------------------------ sample files (a complete "new drug" M004)
SAMPLE_FILES: Dict[str, str] = {
    "sample_medications.csv":
        "drug_id,drug_name,category,demo_status\nM004,Amoxicillin,Antibiotic,SHORTAGE_DEMO\n",
    "sample_inventory.csv":
        "drug_id,drug_name,current_stock,daily_usage,reorder_level,supplier_id,unit_price,last_restock_date\n"
        "M002,,150,,,,,\n"
        "M004,Amoxicillin,40,20,80,S09,1.8,2026-09-20\n",
    "sample_shortage.csv":
        "drug_id,drug_name,shortage_status,reported_date,source,severity,notes\n"
        "M004,Amoxicillin,SHORTAGE,2026-09-27,uploaded-sample,70,Supplier delay reported (sample data)\n",
    "sample_suppliers.csv":
        "supplier_id,supplier_name,medication,available_quantity,unit_price,lead_time_days,supplier_status\n"
        "S09,PharmaLink Demo,Amoxicillin,500,1.7,3,ACTIVE\n",
    "sample_synthetic_patients.csv":
        "patient_id,age_group,condition_category,medication,risk_group,active_status\n"
        "SYN-9001,ADULT,INFECTION,Amoxicillin,HIGH,1\n"
        "SYN-9002,PEDIATRIC,INFECTION,Amoxicillin,MEDIUM,1\n",
    "sample_guideline_amoxicillin.txt":
        "Amoxicillin supply guidance (SAMPLE DOCUMENT)\n\n"
        "During an Amoxicillin shortage, the pharmacy should review stock coverage daily and contact "
        "secondary suppliers. Any alternative product requires qualified pharmacist review before use.\n",
}


def ingest_samples() -> list:
    return [ingest_bytes(n, t.encode("utf-8")) for n, t in SAMPLE_FILES.items()]

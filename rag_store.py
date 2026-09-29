"""
rag_store.py — the ONE unified RAG knowledge base of MedSupply Sentinel (no Streamlit code here).

SINGLE SOURCE OF TRUTH
----------------------
    Data Upload (kb.py: validate + upsert)
        -> persistent storage  : synthetic/*.csv, guidelines/*.pdf, knowledge_base/documents,
                                 knowledge_base/reference_tables, knowledge_base/workflow_context.json
        -> rag_store.sync()    : converts EVERYTHING above into searchable documents with metadata
        -> Unified index       : knowledge_base/rag_index.json (+ in-memory TF-IDF vectors)
        -> chat_engine.Retriever -> LLM -> Chat

The index is never edited by hand: every sync() rebuilds the complete document set from the stored
data and diffs it against the persisted index (add / update / delete by deterministic doc_id).
Old records therefore cannot survive next to new ones (no stale / contradictory answers).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import sentinel_core as core

# ------------------------------------------------------------------ paths
KB_DIR = core.BASE_DIR / "knowledge_base"
DOCS_DIR = KB_DIR / "documents"                 # raw uploaded TXT / MD (UTF-8, Arabic OK)
REFERENCE_DIR = KB_DIR / "reference_tables"     # uploaded tables that are not one of the 5 known tables
INDEX_FILE = KB_DIR / "rag_index.json"          # persisted unified index
WORKFLOW_FILE = KB_DIR / "workflow_context.json"  # persisted results of the multi-agent workflow
UPLOADED_PREFIX = "uploaded_"
for _d in (KB_DIR, DOCS_DIR, REFERENCE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SCHEMA_VERSION = 1

# source types (metadata "source_type") and the provenance group the Chat reports for each
SRC_TABLE = "uploaded_table"        # one of the 5 known tables (medications, inventory, suppliers, ...)
SRC_REFERENCE = "reference_table"   # any other uploaded CSV / Excel
SRC_DERIVED = "knowledge_base"      # joined views built from the tables (medication profile, directories)
SRC_GUIDELINE = "guideline"         # PDFs that came with the dataset
SRC_DOCUMENT = "uploaded_document"  # PDF / TXT / MD uploaded by the user
SRC_WORKFLOW = "workflow"           # output of the LangGraph agents

GROUP_DATA, GROUP_DOCS, GROUP_WORKFLOW = "data", "documents", "workflow"
_GROUP = {SRC_TABLE: GROUP_DATA, SRC_REFERENCE: GROUP_DATA, SRC_DERIVED: GROUP_DATA,
          SRC_GUIDELINE: GROUP_DOCS, SRC_DOCUMENT: GROUP_DOCS, SRC_WORKFLOW: GROUP_WORKFLOW}

_LOCK = threading.RLock()


# ------------------------------------------------------------------ small helpers
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def norm_text(s: Any) -> str:
    """Lower-case + Arabic normalisation (tashkeel, tatweel, alef/yaa/taa-marbuta variants)."""
    s = str(s).lower()
    s = re.sub(r"[\u064B-\u0652\u0640]", "", s)
    s = re.sub(r"[أإآ]", "ا", s).replace("ى", "ي").replace("ة", "ه")
    return s


def urgency_label(coverage: float, stock: int, reorder: int) -> str:
    """Same rule as the Inventory agent in sentinel_core."""
    return ("CRITICAL" if coverage < 2 else
            "HIGH" if coverage < 4 or stock <= reorder else
            "MEDIUM" if coverage < 7 else "LOW")


def _g(x: Any) -> str:
    try:
        return f"{float(x):g}"
    except (TypeError, ValueError):
        return str(x)


def _file_ts(path) -> str:
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    except OSError:
        return now_iso()


def _table_file(name: str) -> str:
    return Path(core._TABLE_SPECS[name]["path"]).name


def _txt(v: Any) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()


# ------------------------------------------------------------------ document model
@dataclass
class Doc:
    doc_id: str
    text: str
    meta: Dict[str, Any]

    @property
    def group(self) -> str:
        return _GROUP.get(self.meta.get("source_type"), GROUP_DATA)

    @property
    def label(self) -> str:
        return f"{self.meta.get('source_file')} → {self.meta.get('display') or self.meta.get('record_id')}"

    def content_hash(self) -> str:
        m = {k: v for k, v in self.meta.items() if k not in ("upload_timestamp", "content_hash")}
        return hashlib.sha1((self.text + json.dumps(m, sort_keys=True, default=str)).encode("utf-8")).hexdigest()

    def to_json(self) -> dict:
        return {"doc_id": self.doc_id, "text": self.text, "meta": self.meta}


def _meta(source_type: str, source_file: str, document_type: str, record_id: str, ts: str, display: str,
          **extra) -> Dict[str, Any]:
    m: Dict[str, Any] = {
        "source_type": source_type, "source_file": source_file, "record_id": record_id,
        "medication_id": None, "medication_name": None, "supplier_id": None, "supplier_name": None,
        "patient_id": None, "document_type": document_type, "upload_timestamp": ts,
        "display": display, "medication_keys": [],
    }
    m.update(extra)
    return m


# ------------------------------------------------------------------ medication identity registry
class Registry:
    """One canonical key per medication. Names in inventory / suppliers / patients that differ slightly from
    the medications table (e.g. 'X 50 mcg' vs 'X') are resolved to the same medication, dynamically."""

    def __init__(self, t: Dict[str, pd.DataFrame]):
        self.names: Dict[str, str] = {}          # key -> display name
        self.ids: Dict[str, str] = {}            # key -> drug_id
        self.variants: Dict[str, set] = {}       # key -> every spelling seen
        for tb in ("medications", "inventory", "shortage_events"):
            for r in t[tb].to_dict("records"):
                self._add(r["drug_name"], r["drug_id"])
        for tb in ("suppliers", "patients"):
            for r in t[tb].to_dict("records"):
                self.resolve(r["medication"], create=True)
        self.key_by_id = {v.lower(): k for k, v in self.ids.items()}
        self._norm_variants = {k: [norm_text(v) for v in vs if len(norm_text(v)) >= 4]
                               for k, vs in self.variants.items()}

    def _add(self, name: Any, did: Any = "") -> Optional[str]:
        n = _txt(name)
        if not n:
            return None
        key = n.lower()
        self.names.setdefault(key, n)
        self.variants.setdefault(key, set()).add(key)
        if _txt(did):
            self.ids.setdefault(key, _txt(did))
        return key

    def resolve(self, name: Any, create: bool = False) -> Optional[str]:
        key = _txt(name).lower()
        if not key:
            return None
        if key in self.names:
            return key
        nk = norm_text(key)
        inside = [k for k in self.names if len(norm_text(k)) >= 4 and norm_text(k) in nk]
        if inside:                                   # 'levothyroxine 50 mcg' contains 'levothyroxine'
            best = max(inside, key=len)
            self.variants[best].add(key)
            return best
        outside = [k for k in self.names if len(nk) >= 4 and nk in norm_text(k)]
        if len(outside) == 1:                        # 'insulin' -> the only 'insulin glargine'
            self.variants[outside[0]].add(key)
            return outside[0]
        return self._add(name) if create else None

    def mentions(self, text: str) -> List[str]:
        nt = norm_text(text)
        return [k for k, vs in self._norm_variants.items() if any(v in nt for v in vs)]


# ------------------------------------------------------------------ chunking / reading documents
def chunk_text(text: str, size: int = 1100, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out, i = [], 0
    while i < len(text):
        end = min(i + size, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", i + size // 2, end), text.rfind("؟ ", i + size // 2, end),
                      text.rfind("。", i + size // 2, end))
            if cut > i:
                end = cut + 1
        piece = text[i:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        i = max(end - overlap, i + 1)
    return out


@lru_cache(maxsize=256)
def _pdf_pages(path_str: str, mtime: float, size: int) -> tuple:
    reader = PdfReader(path_str)
    return tuple((i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1))


def _derived_stems() -> set:
    """Stems of PDFs that kb.py generated from a TXT/MD/table (the RAG indexes the raw source instead)."""
    stems = {p.stem for p in DOCS_DIR.glob("*") if p.suffix.lower() in (".txt", ".md")}
    stems |= {p.stem for p in REFERENCE_DIR.glob("*.csv")}
    return stems


def guideline_pdfs() -> List[Tuple[str, Path]]:
    """[(origin, path)] for every PDF the RAG should index; origin = 'dataset' | 'uploaded'."""
    d = Path(core.GUIDELINE_DIR)
    if not d.is_dir():
        return []
    skip = _derived_stems()
    return [("uploaded" if f.name.startswith(UPLOADED_PREFIX) else "dataset", f)
            for f in sorted(d.glob("*.pdf")) if f.stem not in skip]


def sources_fingerprint() -> tuple:
    """Changes whenever any stored source (table, PDF, raw doc, reference table, workflow file) changes."""
    paths = [Path(core._TABLE_SPECS[n]["path"]) for n in core._TABLE_SPECS]
    paths += [f for _, f in guideline_pdfs()]
    paths += sorted(DOCS_DIR.glob("*")) + sorted(REFERENCE_DIR.glob("*.csv")) + [WORKFLOW_FILE]
    fp = [SCHEMA_VERSION]
    for p in paths:
        try:
            st = p.stat()
            fp.append((p.name, st.st_mtime_ns, st.st_size))
        except OSError:
            fp.append((p.name, 0, 0))
    return tuple(fp)


def tables_fingerprint() -> str:
    parts = []
    for n in core._TABLE_SPECS:
        try:
            st = Path(core._TABLE_SPECS[n]["path"]).stat()
            parts.append(f"{n}:{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            parts.append(f"{n}:0:0")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


# ------------------------------------------------------------------ building documents
def _load_tables() -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    out, warns = {}, []
    for n, spec in core._TABLE_SPECS.items():
        try:
            out[n] = core.load_local_table(n)
        except Exception as exc:                      # missing / invalid file -> empty table + warning
            out[n] = pd.DataFrame(columns=spec["columns"])
            warns.append(f"{_table_file(n)}: {type(exc).__name__}: {exc}")
    return out, warns


def _text_docs(reg: Registry) -> List[Doc]:
    """Guideline PDFs, uploaded PDFs, uploaded TXT/MD and uploaded reference tables -> chunks."""
    docs: List[Doc] = []
    for origin, f in guideline_pdfs():
        stype = SRC_DOCUMENT if origin == "uploaded" else SRC_GUIDELINE
        dtype = "document_chunk" if origin == "uploaded" else "guideline_chunk"
        ts = _file_ts(f)
        try:
            pages = _pdf_pages(str(f), f.stat().st_mtime, f.stat().st_size)
        except Exception:
            continue
        for num, page_text in pages:
            for k, chunk in enumerate(chunk_text(page_text), start=1):
                keys = reg.mentions(chunk)
                m = _meta(stype, f.name, dtype, f"{f.name}#p{num}c{k}", ts, f"{f.name} p.{num}",
                          page=num, medication_keys=keys,
                          medication_name=reg.names[keys[0]] if keys else None,
                          medication_id=reg.ids.get(keys[0]) if keys else None)
                docs.append(Doc(f"{f.name}#p{num}c{k}", chunk, m))
    for f in sorted(DOCS_DIR.glob("*")):
        if f.suffix.lower() not in (".txt", ".md"):
            continue
        ts = _file_ts(f)
        raw = f.read_text(encoding="utf-8", errors="ignore")
        for k, chunk in enumerate(chunk_text(raw), start=1):
            keys = reg.mentions(chunk)
            m = _meta(SRC_DOCUMENT, f.name, "document_chunk", f"{f.name}#c{k}", ts, f"{f.name} part {k}",
                      medication_keys=keys, medication_name=reg.names[keys[0]] if keys else None,
                      medication_id=reg.ids.get(keys[0]) if keys else None)
            docs.append(Doc(f"{f.name}#c{k}", chunk, m))
    for f in sorted(REFERENCE_DIR.glob("*.csv")):
        ts = _file_ts(f)
        try:
            df = pd.read_csv(f, dtype=str, encoding="utf-8").fillna("")
        except Exception:
            continue
        for n, row in enumerate(df.to_dict("records"), start=1):
            cells = [f"{c}: {v}" for c, v in row.items() if str(v).strip()]
            if not cells:
                continue
            text = f"Reference table {f.name} — row {n}\n" + "\n".join(cells)
            keys = reg.mentions(text)
            m = _meta(SRC_REFERENCE, f.name, "reference_row", f"{f.name}#row{n}", ts,
                      " | ".join(cells)[:80], medication_keys=keys,
                      medication_name=reg.names[keys[0]] if keys else None,
                      medication_id=reg.ids.get(keys[0]) if keys else None)
            docs.append(Doc(f"{f.name}#row{n}", text, m))
    return docs


def _table_docs(t: Dict[str, pd.DataFrame], reg: Registry, text_docs: List[Doc]) -> List[Doc]:
    """Structured rows -> meaningful, relationship-aware documents."""
    docs: List[Doc] = []
    f = {n: _table_file(n) for n in t}
    ts = {n: _file_ts(core._TABLE_SPECS[n]["path"]) for n in t}
    med, inv, sup, ev, pat = (t[n].copy() for n in ("medications", "inventory", "suppliers",
                                                     "shortage_events", "patients"))
    med["_key"] = med["drug_name"].map(reg.resolve)
    inv["_key"] = inv["drug_name"].map(reg.resolve)
    ev["_key"] = ev["drug_name"].map(reg.resolve)
    sup["_key"] = sup["medication"].map(reg.resolve)
    pat["_key"] = pat["medication"].map(reg.resolve)

    # ---- per-medication lookups (used for relationship lines)
    inv_by = {r["_key"]: r for r in inv.to_dict("records")}
    ev_all: Dict[str, List[dict]] = {}
    event_rows = ev.to_dict("records")
    for r in event_rows:
        ev_all.setdefault(r["_key"], []).append(r)
    latest_ev: Dict[str, dict] = {}
    for k, rows in ev_all.items():                                  # newest date wins; ties -> later row
        order = sorted(range(len(rows)), key=lambda i: (pd.to_datetime(rows[i]["reported_date"], errors="coerce"), i))
        latest_ev[k] = rows[order[-1]]
        for r in rows:
            r["_latest"] = r is latest_ev[k]
    sup_by: Dict[str, List[dict]] = {}
    for r in sup.to_dict("records"):
        sup_by.setdefault(r["_key"], []).append(r)
    for k in sup_by:
        sup_by[k].sort(key=lambda r: (r["supplier_status"] != "ACTIVE", r["unit_price"]))
    sup_name = {}
    for r in sup.to_dict("records"):
        sup_name.setdefault(r["supplier_id"], r["supplier_name"])
    pat_by: Dict[str, pd.DataFrame] = {k: g for k, g in pat.groupby("_key")} if len(pat) else {}

    def cov_of(r: dict) -> float:
        return float(r["current_stock"]) / max(float(r["daily_usage"]), 1e-9)

    def related(k: str) -> str:
        bits = []
        if k in latest_ev:
            bits.append(f"shortage status {latest_ev[k]['shortage_status']} (reported {latest_ev[k]['reported_date']})")
        if k in inv_by:
            r = inv_by[k]
            bits.append(f"stock {int(r['current_stock'])} units, coverage {cov_of(r):.2f} days")
        act = [s for s in sup_by.get(k, []) if s["supplier_status"] == "ACTIVE"]
        bits.append(f"{len(act)} active supplier(s)")
        return "Related (same medication): " + "; ".join(bits)

    # ---- medications
    for r in med.to_dict("records"):
        k = r["_key"]
        text = (f"Medication: {r['drug_name']}\nMedication ID: {r['drug_id']}\nCategory: {r['category']}\n"
                f"Catalog status tag: {r['demo_status']}\n{related(k)}")
        docs.append(Doc(f"medications:{r['drug_id']}", text, _meta(
            SRC_TABLE, f["medications"], "medication_record", r["drug_id"], ts["medications"], r["drug_name"],
            medication_id=r["drug_id"], medication_name=r["drug_name"], medication_keys=[k],
            table="medications")))

    # ---- inventory
    for r in inv.to_dict("records"):
        k, cov = r["_key"], cov_of(r)
        urg = urgency_label(cov, int(r["current_stock"]), int(r["reorder_level"]))
        sname = sup_name.get(r["supplier_id"])
        text = (f"Inventory record: {r['drug_name']} ({r['drug_id']})\n"
                f"Current stock: {int(r['current_stock'])} units\nDaily usage: {_g(r['daily_usage'])} units/day\n"
                f"Coverage days: {cov:.2f}\nReorder level: {int(r['reorder_level'])} units\n"
                f"At or below reorder level: {'yes' if int(r['current_stock']) <= int(r['reorder_level']) else 'no'}\n"
                f"Urgency: {urg}\nPrimary supplier: {sname + ' ' if sname else ''}({r['supplier_id']})\n"
                f"Unit price: {_g(r['unit_price'])}\nLast restock date: {r['last_restock_date']}\n{related(k)}")
        docs.append(Doc(f"inventory:{r['drug_id']}", text, _meta(
            SRC_TABLE, f["inventory"], "inventory_record", r["drug_id"], ts["inventory"],
            f"{r['drug_name']}: current stock {int(r['current_stock'])} units, {cov:.1f} days",
            medication_id=r["drug_id"], medication_name=r["drug_name"], supplier_id=r["supplier_id"],
            supplier_name=sname, medication_keys=[k], urgency=urg, coverage_days=round(cov, 2),
            table="inventory")))

    # ---- suppliers
    for r in sup.to_dict("records"):
        k = r["_key"]
        text = (f"Supplier: {r['supplier_name']}\nSupplier ID: {r['supplier_id']}\nStatus: {r['supplier_status']}\n"
                f"Medication: {r['medication']}\nMedication ID: {reg.ids.get(k, 'n/a')}\n"
                f"Availability: {int(r['available_quantity'])} units available\nUnit price: {_g(r['unit_price'])}\n"
                f"Lead time: {int(r['lead_time_days'])} days\n{related(k)}")
        docs.append(Doc(f"suppliers:{r['supplier_id']}|{r['medication']}", text, _meta(
            SRC_TABLE, f["suppliers"], "supplier_record", f"{r['supplier_id']}|{r['medication']}",
            ts["suppliers"], f"{r['supplier_name']} — {r['medication']}",
            medication_id=reg.ids.get(k), medication_name=reg.names.get(k), supplier_id=r["supplier_id"],
            supplier_name=r["supplier_name"], medication_keys=[k], supplier_status=r["supplier_status"],
            unit_price=float(r["unit_price"]), available_quantity=int(r["available_quantity"]),
            table="suppliers")))

    # ---- shortage events
    for r in event_rows:
        k = r["_key"]
        text = (f"Shortage event: {r['drug_name']} ({r['drug_id']})\nShortage status: {r['shortage_status']}\n"
                f"Severity: {int(r['severity'])}/100\nReported date: {r['reported_date']}\nSource: {r['source']}\n"
                + (f"Notes: {r['notes']}\n" if _txt(r["notes"]) else "")
                + f"Latest report for this medication (= current status): {'yes' if r['_latest'] else 'no, superseded by a newer report'}\n"
                + related(k))
        docs.append(Doc(f"shortage_events:{r['drug_id']}|{r['reported_date']}", text, _meta(
            SRC_TABLE, f["shortage_events"], "shortage_event", f"{r['drug_id']}|{r['reported_date']}",
            ts["shortage_events"], f"{r['drug_name']}: {r['shortage_status']} ({r['reported_date']})",
            medication_id=r["drug_id"], medication_name=r["drug_name"], medication_keys=[k],
            shortage_status=r["shortage_status"], is_latest=bool(r["_latest"]),
            reported_date=r["reported_date"], table="shortage_events")))

    # ---- synthetic patients (rows + one impact summary per medication)
    for r in pat.to_dict("records"):
        k = r["_key"]
        text = (f"Synthetic patient: {r['patient_id']}\nAge group: {r['age_group']}\n"
                f"Condition category: {r['condition_category']}\nMedication: {r['medication']}\n"
                f"Medication ID: {reg.ids.get(k, 'n/a')}\nRisk group: {r['risk_group']}\n"
                f"Active: {'yes' if int(r['active_status']) == 1 else 'no'}")
        docs.append(Doc(f"patients:{r['patient_id']}", text, _meta(
            SRC_TABLE, f["patients"], "patient_record", r["patient_id"], ts["patients"],
            f"{r['patient_id']} ({r['condition_category']}, {r['risk_group']})",
            medication_id=reg.ids.get(k), medication_name=reg.names.get(k), patient_id=r["patient_id"],
            medication_keys=[k], table="patients")))

    # ---- guideline references per medication (relationship: medication <-> guideline)
    refs: Dict[str, List[str]] = {}
    for d in text_docs:
        for k in d.meta.get("medication_keys", []):
            label = d.meta["display"]
            if label not in refs.setdefault(k, []):
                refs[k].append(label)

    # ---- joined medication profiles (+ patient summaries)
    all_keys = sorted(reg.names)
    overview: List[str] = []
    for k in all_keys:
        name, did = reg.names[k], reg.ids.get(k, "")
        lines = [f"MEDICATION PROFILE — {name}" + (f" (drug_id {did})" if did else "")]
        mrow = med[med["_key"] == k]
        if not mrow.empty:
            lines.append(f"Category: {mrow.iloc[0]['category']}; catalog tag: {mrow.iloc[0]['demo_status']}")
        status, sev = "NO EVENT RECORDED", 0
        if k in latest_ev:
            e = latest_ev[k]
            status, sev = e["shortage_status"], int(e["severity"])
            lines.append(f"Shortage status (latest report): {status} — severity {sev}/100, reported "
                         f"{e['reported_date']}, source {e['source']}" + (f", notes: {e['notes']}" if _txt(e["notes"]) else "")
                         + f". Reports on file: {len(ev_all[k])}")
        else:
            lines.append("Shortage status: no shortage event recorded.")
        stock_txt, urg, cov, below = "no inventory record", "", None, False
        if k in inv_by:
            i = inv_by[k]
            cov = cov_of(i)
            below = int(i["current_stock"]) <= int(i["reorder_level"])
            urg = urgency_label(cov, int(i["current_stock"]), int(i["reorder_level"]))
            stock_txt = f"stock={int(i['current_stock'])}, coverage={cov:.1f}d"
            sname = sup_name.get(i["supplier_id"])
            lines.append(f"Inventory: stock {int(i['current_stock'])} units, daily usage {_g(i['daily_usage'])}, "
                         f"coverage_days {cov:.2f}, reorder level {int(i['reorder_level'])}, urgency {urg}, "
                         f"primary supplier {sname + ' ' if sname else ''}({i['supplier_id']}), unit price "
                         f"{_g(i['unit_price'])}, last restock {i['last_restock_date']}")
        else:
            lines.append("Inventory: no inventory record.")
        sl = sup_by.get(k, [])
        act = [s for s in sl if s["supplier_status"] == "ACTIVE"]
        if sl:
            lines.append(f"Suppliers ({len(sl)} on file, {len(act)} active): " + "; ".join(
                f"{s['supplier_name']} ({s['supplier_id']}, {s['supplier_status']}) qty {int(s['available_quantity'])}, "
                f"price {_g(s['unit_price'])}, lead time {int(s['lead_time_days'])}d" for s in sl[:8]))
            if act:
                lines.append("Candidate supplier options (active suppliers by lowest unit price; operational "
                             "information only, pharmacist review required): " + "; ".join(
                                 f"{n}) {s['supplier_name']}" for n, s in enumerate(sorted(act, key=lambda s: s["unit_price"])[:3], 1)))
        else:
            lines.append("Suppliers: no supplier records.")
        pg = pat_by.get(k)
        n_pat = 0
        if pg is not None:
            active = pg[pg["active_status"] == 1]
            n_pat = len(active)
            if n_pat:
                cats = ", ".join(f"{c}: {v}" for c, v in active["condition_category"].value_counts().items())
                high = int((active["risk_group"] == "HIGH").sum())
                lines.append(f"Active synthetic patients: {n_pat} (HIGH risk: {high}); categories: {cats}")
                docs.append(Doc(f"patients_summary:{k}", (
                    f"Synthetic patient impact — {name}\nActive synthetic patients: {n_pat}\nHIGH risk: {high}\n"
                    f"Categories: {cats}\nTotal patient records (incl. inactive): {len(pg)}"), _meta(
                    SRC_DERIVED, f["patients"], "patient_impact_summary", k, ts["patients"],
                    f"{name}: {n_pat} active synthetic patients", medication_id=did or None, medication_name=name,
                    medication_keys=[k], table="patients")))
            else:
                lines.append("Active synthetic patients: 0.")
        else:
            lines.append("Active synthetic patients: none recorded.")
        lines.append("Guideline / document references: " + ("; ".join(refs[k][:6]) if refs.get(k)
                                                             else "no guideline or uploaded document mentions this medication."))
        docs.append(Doc(f"profile:{k}", "\n".join(lines), _meta(
            SRC_DERIVED, "joined view (medications + inventory + suppliers + shortage_events + patients + guidelines)",
            "medication_profile", k, now_iso(), f"{name} (joined profile)",
            medication_id=did or None, medication_name=name, medication_keys=[k],
            name_variants=sorted(reg.variants.get(k, {k})), current_shortage_status=status,
            severity=sev, urgency=urg or None, coverage_days=None if cov is None else round(cov, 2),
            below_reorder=below, active_supplier_count=len(act), supplier_ids=sorted({s["supplier_id"] for s in sl}),
            supplier_names=sorted({s["supplier_name"] for s in sl}), patient_count=n_pat)))
        overview.append(f"- {name} ({did or 'n/a'}): shortage={status}; {stock_txt}; urgency={urg or 'n/a'}; "
                        f"active_suppliers={len(act)}; active_synthetic_patients={n_pat}")

    for i in range(0, len(overview), 30):
        docs.append(Doc(f"overview:{i // 30}", "OVERVIEW OF ALL MEDICATIONS IN THE KNOWLEDGE BASE:\n"
                        + "\n".join(overview[i:i + 30]), _meta(
            SRC_DERIVED, "joined view (all tables)", "overview", f"overview-{i // 30}", now_iso(),
            "all medications overview")))
    dirs = [f"- {r['supplier_name']} ({r['supplier_id']}, {r['supplier_status']}): {r['medication']} — "
            f"qty {int(r['available_quantity'])}, price {_g(r['unit_price'])}, lead time {int(r['lead_time_days'])}d"
            for r in sorted(sup.to_dict("records"), key=lambda r: (r["supplier_name"], r["medication"]))]
    for i in range(0, len(dirs), 30):
        docs.append(Doc(f"directory:suppliers:{i // 30}", "SUPPLIER DIRECTORY (all supplier records):\n"
                        + "\n".join(dirs[i:i + 30]), _meta(
            SRC_DERIVED, f["suppliers"], "supplier_directory", f"directory-{i // 30}", ts["suppliers"],
            "supplier directory")))
    return docs


# ------------------------------------------------------------------ workflow context
def _load_workflows() -> Dict[str, dict]:
    if WORKFLOW_FILE.is_file():
        try:
            return json.loads(WORKFLOW_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def record_workflow(state: dict, stage: str = "COMPLETED") -> Optional[dict]:
    """Persist the (partial or final) LangGraph state so the Chat can retrieve and explain it.
    Called by app.py when the workflow pauses at the pharmacist gate and again when it finishes."""
    sd = state.get("shortage_data") or {}
    wid = state.get("workflow_id")
    if not wid or not sd.get("drug_id"):
        return None
    gd = state.get("guideline_data") or {}
    snap = {
        "workflow_id": wid, "stage": stage, "recorded_at": now_iso(), "data_fingerprint": tables_fingerprint(),
        "drug_id": sd.get("drug_id"), "drug_name": sd.get("drug_name"),
        "shortage_data": {k: v for k, v in sd.items() if k != "evidence"},
        "inventory_data": {k: v for k, v in (state.get("inventory_data") or {}).items() if k != "evidence"},
        "guideline_data": {"documents_found": gd.get("documents_found", []), "constraints": gd.get("constraints", []),
                           "applicable_guidance": [g[:400] for g in gd.get("applicable_guidance", [])[:3]]},
        "patient_impact": {k: v for k, v in (state.get("patient_impact") or {}).items() if k != "evidence"},
        "candidate_alternatives": {k: v for k, v in (state.get("candidate_alternatives") or {}).items()
                                   if k != "supporting_evidence"},
        "validation_result": state.get("validation_result"),
        "approval_status": state.get("approval_status"), "reviewer_role": state.get("reviewer_role"),
        "decision_reason": state.get("decision_reason"), "purchase_order": state.get("purchase_order"),
        "final_report": (state.get("final_report") or "")[:6000],
    }
    with _LOCK:
        allw = _load_workflows()
        allw[wid] = json.loads(json.dumps(snap, default=str))
        newest = sorted(allw.values(), key=lambda s: s.get("recorded_at", ""), reverse=True)[:50]
        WORKFLOW_FILE.write_text(json.dumps({s["workflow_id"]: s for s in newest}, indent=1, ensure_ascii=False),
                                 encoding="utf-8")
        get_kb().sync(force=True)
    return snap


def _workflow_docs(reg: Registry) -> List[Doc]:
    docs: List[Doc] = []
    latest: Dict[str, dict] = {}                       # newest snapshot per medication only
    for s in _load_workflows().values():
        k = reg.key_by_id.get(str(s.get("drug_id", "")).lower()) or reg.resolve(s.get("drug_name"))
        if k is None:
            continue                                   # medication no longer exists in the data
        s["_key"] = k
        if k not in latest or s.get("recorded_at", "") > latest[k].get("recorded_at", ""):
            latest[k] = s
    cur_fp = tables_fingerprint()
    for k, s in latest.items():
        wid, name, did = s["workflow_id"], reg.names[k], reg.ids.get(k)
        stale = s.get("data_fingerprint") != cur_fp
        head = (f"WORKFLOW RESULT (generated by the LangGraph agents, stage {s.get('stage')}, workflow {wid}, "
                f"recorded {s.get('recorded_at')}). This is AI/derived analysis, not raw uploaded data.\n"
                + ("NOTE: uploaded data changed after this workflow ran — re-run the workflow to refresh it.\n"
                   if stale else "") + f"Medication: {name}" + (f" ({did})" if did else ""))

        def add(kind: str, title: str, body: str):
            docs.append(Doc(f"workflow:{wid}:{kind}", f"{head}\n{title}\n{body}", _meta(
                SRC_WORKFLOW, f"workflow/{wid}", f"workflow_{kind}", f"{wid}:{kind}", s.get("recorded_at", now_iso()),
                f"{title} ({wid})", medication_id=did, medication_name=name, medication_keys=[k],
                workflow_id=wid, workflow_stage=s.get("stage"), stale=stale)))

        sd, inv = s.get("shortage_data") or {}, s.get("inventory_data") or {}
        if sd:
            add("shortage_analysis", "Shortage Monitor Agent", (
                f"Shortage status: {sd.get('shortage_status')}\nSeverity: {sd.get('severity')}/100\n"
                f"Reported date: {sd.get('reported_date')}\nSource: {sd.get('source')}\nNotes: {sd.get('notes', '')}"))
        if inv:
            add("inventory_analysis", "Inventory Analyst Agent", (
                f"Current stock: {inv.get('current_stock')} units\nDaily usage: {inv.get('daily_usage')}\n"
                f"Coverage days: {inv.get('coverage_days')}\nCalculation: {inv.get('calculation')}\n"
                f"Urgency: {inv.get('urgency')}\nSufficient stock: {inv.get('sufficient_stock')}\n"
                f"Reorder level: {inv.get('reorder_level')}"))
        gd = s.get("guideline_data") or {}
        if gd.get("documents_found") or gd.get("constraints"):
            add("guideline_analysis", "Guideline Reader Agent", (
                f"Documents found: {', '.join(gd.get('documents_found', []))}\n"
                f"Constraints: {'; '.join(gd.get('constraints', [])) or 'none recorded'}\n"
                + "\n".join(f"Guidance excerpt: {g}" for g in gd.get("applicable_guidance", []))))
        pi = s.get("patient_impact") or {}
        if pi:
            add("patient_impact", "Patient Impact Analyst Agent", (
                f"Affected synthetic patients: {pi.get('affected_patient_count')}\n"
                f"Categories: {', '.join(pi.get('affected_categories', []))}\n"
                f"High-priority cases: {pi.get('high_priority_cases')}\nOperational urgency: {pi.get('operational_urgency')}\n"
                f"Reasoning: {pi.get('reasoning', '')}"))
        ca = s.get("candidate_alternatives") or {}
        if ca:
            cands = ca.get("candidate_alternatives", [])
            add("substitution_analysis", "Substitution Analyst Agent", (
                (" ; ".join(f"{c.get('supplier_name')}: qty {c.get('available_quantity')}, price {c.get('unit_price')}, "
                            f"lead time {c.get('lead_time_days')}d" for c in cands) or "No candidate options were found.")
                + f"\nConstraints: {'; '.join(ca.get('constraints', [])) or 'none'}\n"
                  f"Uncertainties: {'; '.join(ca.get('uncertainties', []))}\nPharmacist review required: yes"))
        vr = s.get("validation_result") or {}
        decision = s.get("approval_status")
        decided = s.get("stage") == "COMPLETED"
        add("validation", "Pharmacist Validator and human approval", (
            f"Validator status: {vr.get('status', 'n/a')}\nValidator reasons: {'; '.join(vr.get('reasons', []))}\n"
            + (f"Human decision: {decision} by {s.get('reviewer_role') or 'n/a'}; reason: {s.get('decision_reason') or 'n/a'}"
               if decided else "Human decision: still awaiting pharmacist review (no approval yet, no purchase order).")))
        po = s.get("purchase_order")
        if po:
            add("purchase_order", "Purchase order draft", (
                f"PO number: {po.get('po_number')}\nSupplier: {po.get('supplier')}\nQuantity: {po.get('quantity')}\n"
                f"Unit price: {po.get('unit_price')}\nEstimated total: {po.get('estimated_total')}\n"
                f"Status: {po.get('status')}"))
        for n, chunk in enumerate(chunk_text(s.get("final_report") or "", size=1500, overlap=100)[:3], start=1):
            add(f"report_{n}", f"Final incident report part {n}", chunk)
    return docs


def collect_docs() -> Tuple[List[Doc], List[str]]:
    t, warns = _load_tables()
    reg = Registry(t)
    text_docs = _text_docs(reg)
    docs = _table_docs(t, reg, text_docs) + text_docs + _workflow_docs(reg)
    return docs, warns


# ------------------------------------------------------------------ the knowledge base
@dataclass
class SyncReport:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    total: int = 0
    skipped: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)

    def summary(self) -> str:
        return (f"+{self.added} added, ~{self.updated} updated, -{self.removed} removed "
                f"({self.total} documents in the index)")

    def as_dict(self) -> dict:
        return {"added": self.added, "updated": self.updated, "removed": self.removed,
                "unchanged": self.unchanged, "total": self.total, "warnings": self.warnings}


class KnowledgeBase:
    def __init__(self):
        self.docs: Dict[str, Doc] = {}
        self.version = 0
        self._fp: Optional[tuple] = None
        self._vec_version = -1
        self._ent_version = -1
        self._ent: dict = {}
        self._ids: List[str] = []
        self._wv = self._cv = self._wm = self._cm = None
        self.vocab: set = set()
        self.prefixes: set = set()
        self._load()

    # ---- persistence
    def _load(self) -> None:
        try:
            raw = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
            if raw.get("schema") == SCHEMA_VERSION:
                self.docs = {d["doc_id"]: Doc(d["doc_id"], d["text"], d["meta"]) for d in raw["docs"]}
                self._fp = tuple(tuple(x) if isinstance(x, list) else x for x in raw.get("fingerprint", []))
                self.version = 1
        except Exception:
            pass

    def _save(self) -> None:
        tmp = INDEX_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"schema": SCHEMA_VERSION, "synced_at": now_iso(), "fingerprint": self._fp,
                                   "docs": [d.to_json() for d in self.docs.values()]},
                                  ensure_ascii=False), encoding="utf-8")
        tmp.replace(INDEX_FILE)

    # ---- sync = ingestion / re-indexing
    def sync(self, force: bool = False) -> SyncReport:
        """Rebuild all documents from the stored data and reconcile the index (upsert changed, delete gone)."""
        with _LOCK:
            fp = sources_fingerprint()
            if not force and self.version and self._fp == fp:
                return SyncReport(unchanged=len(self.docs), total=len(self.docs), skipped=True)
            new_docs, warns = collect_docs()
            rep = SyncReport(warnings=warns)
            fresh: Dict[str, Doc] = {}
            for d in new_docs:
                h = d.content_hash()
                old = self.docs.get(d.doc_id)
                if old is None:
                    rep.added += 1
                elif old.meta.get("content_hash") != h:
                    rep.updated += 1
                else:
                    rep.unchanged += 1
                    d.meta["upload_timestamp"] = old.meta.get("upload_timestamp", d.meta["upload_timestamp"])
                d.meta["content_hash"] = h
                fresh[d.doc_id] = d
            rep.removed = len([i for i in self.docs if i not in fresh])
            rep.total = len(fresh)
            self.docs, self._fp = fresh, fp
            self.version += 1
            self._save()
            return rep

    def sync_if_needed(self) -> SyncReport:
        return self.sync(force=False)

    # ---- metadata lookup
    def get(self, doc_id: str) -> Optional[Doc]:
        return self.docs.get(doc_id)

    def find(self, **conds) -> List[Doc]:
        """Metadata filter. Scalar meta fields compare with ==, list fields (medication_keys) with membership."""
        out = []
        for d in self.docs.values():
            for k, want in conds.items():
                have = d.meta.get(k)
                ok = (want in have) if isinstance(have, (list, tuple, set)) else (have == want)
                if not ok:
                    break
            else:
                out.append(d)
        return out

    def entities(self) -> dict:
        if self._ent_version == self.version:
            return self._ent
        drugs, sups, pats, ids = [], {}, [], set()
        for p in self.find(document_type="medication_profile"):
            m = p.meta
            drugs.append({"key": m["medication_keys"][0], "name": m["medication_name"], "id": m.get("medication_id") or "",
                          "variants": m.get("name_variants", [])})
            if m.get("medication_id"):
                ids.add(m["medication_id"].lower())
        for d in self.find(document_type="supplier_record"):
            sups.setdefault(d.meta["supplier_id"], d.meta["supplier_name"])
            ids.add(str(d.meta["supplier_id"]).lower())
        for d in self.find(document_type="patient_record"):
            pats.append(d.meta["patient_id"])
        self._ent = {"drugs": drugs, "suppliers": sups, "patients": {p.lower(): p for p in pats}, "ids": ids}
        self._ent_version = self.version
        return self._ent

    def drug_name(self, key: str) -> str:
        for d in self.entities()["drugs"]:
            if d["key"] == key:
                return d["name"]
        return key

    # ---- semantic layer (TF-IDF over words + character n-grams; no external model or network needed)
    def _ensure_vectors(self) -> None:
        if self._vec_version == self.version:
            return
        self._ids = list(self.docs)
        corpus = [f"{self.docs[i].meta.get('display', '')}\n{self.docs[i].text}" for i in self._ids]
        self._wv = self._cv = self._wm = self._cm = None
        self.vocab, self.prefixes = set(), set()
        if corpus:
            self._wv = TfidfVectorizer(preprocessor=norm_text, ngram_range=(1, 2), sublinear_tf=True)
            self._wm = self._wv.fit_transform(corpus)
            self._cv = TfidfVectorizer(preprocessor=norm_text, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
            self._cm = self._cv.fit_transform(corpus)
            self.vocab = {w for w in self._wv.vocabulary_ if " " not in w}
            self.prefixes = {w[:5] for w in self.vocab if len(w) >= 5}
        self._vec_version = self.version

    def semantic(self, query: str) -> Dict[str, float]:
        """doc_id -> similarity in [0,1] (0.6 word-level + 0.4 character-level TF-IDF cosine)."""
        with _LOCK:
            self._ensure_vectors()
            if self._wm is None or not query.strip():
                return {}
            sw = linear_kernel(self._wv.transform([query]), self._wm).ravel()
            sc = linear_kernel(self._cv.transform([query]), self._cm).ravel()
            s = 0.6 * sw + 0.4 * sc
            return {i: float(s[n]) for n, i in enumerate(self._ids)}

    def known_word(self, w: str) -> bool:
        with _LOCK:
            self._ensure_vectors()
            w = norm_text(w)
            return w in self.vocab or (len(w) >= 5 and w[:5] in self.prefixes)

    # ---- reporting
    def stats(self) -> dict:
        by_file: Dict[Tuple[str, str, str], int] = {}
        for d in self.docs.values():
            key = (d.meta["source_file"], d.meta["source_type"], d.meta["document_type"])
            by_file[key] = by_file.get(key, 0) + 1
        return {"total": len(self.docs), "version": self.version, "by_file": by_file}

    def stats_frame(self) -> pd.DataFrame:
        rows = [{"source_file": f, "source_type": s, "document_type": t, "documents": n}
                for (f, s, t), n in sorted(self.stats()["by_file"].items())]
        return pd.DataFrame(rows, columns=["source_file", "source_type", "document_type", "documents"])


_KB: Optional[KnowledgeBase] = None


def get_kb() -> KnowledgeBase:
    global _KB
    with _LOCK:
        if _KB is None:
            _KB = KnowledgeBase()
        return _KB


def reset_singleton() -> None:
    """Drop the in-memory instance (tests / after wiping the knowledge_base folder)."""
    global _KB
    with _LOCK:
        _KB = None
"""
MedSupply Sentinel — core (extracted from the Kaggle notebook, LLM layer = Groq API).
Agents / schemas / graph logic are the notebook's, unchanged except: paths, LLM client, graph builder.
"""
from __future__ import annotations

import os
import re
import json
import math
import time
import uuid
import random
import operator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Annotated, TypedDict, Optional, Dict, List

import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from pypdf import PdfReader
from sqlalchemy import create_engine, text as sql_text

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

random.seed(42)

# ---------------------------------------------------------------
# Paths (env-configurable so it runs locally and on Streamlit Cloud)
# ---------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
_DATA_ROOT = Path(os.getenv("MEDSUPPLY_DATA_DIR", APP_DIR / "data"))
BASE_DIR = Path(os.getenv("MEDSUPPLY_OUTPUT_DIR", APP_DIR / "output"))
REPORT_DIR = BASE_DIR / "reports"
NOTIFICATION_DIR = BASE_DIR / "notifications"
DB_PATH = BASE_DIR / "medsupply_sentinel.db"
for _d in (BASE_DIR, REPORT_DIR, NOTIFICATION_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _resolve_dataset_base_dir(root: Path) -> Path:
    """Find the folder that contains synthetic/inventory.csv (root itself or up to 3 levels below)."""
    if (root / "synthetic" / "inventory.csv").is_file():
        return root
    if root.is_dir():
        for cand in sorted(root.rglob("inventory.csv")):
            if cand.parent.name == "synthetic":
                return cand.parent.parent
    return root


DATASET_BASE_DIR = _resolve_dataset_base_dir(_DATA_ROOT)
SYNTHETIC_DIR = DATASET_BASE_DIR / "synthetic"
GUIDELINE_DIR = DATASET_BASE_DIR / "guidelines"
GUIDELINES_DIR = GUIDELINE_DIR
CONFIG_DIR = DATASET_BASE_DIR / "config"

PATIENTS_CSV = SYNTHETIC_DIR / "synthetic_patients.csv"
MEDICATIONS_CSV = SYNTHETIC_DIR / "medications.csv"
INVENTORY_CSV = SYNTHETIC_DIR / "inventory.csv"
SUPPLIERS_CSV = SYNTHETIC_DIR / "suppliers.csv"
SHORTAGE_EVENTS_CSV = SYNTHETIC_DIR / "shortage_events.csv"
CONFIG_FILE = CONFIG_DIR / "demo_scenario.json"

# Expected columns + numeric typing for every local table (column names are NOT renamed)
_TABLE_SPECS: Dict[str, Dict[str, Any]] = {
    "patients": {
        "path": PATIENTS_CSV,
        "columns": ["patient_id", "age_group", "condition_category", "medication", "risk_group", "active_status"],
        "int": ["active_status"], "float": [],
    },
    "medications": {
        "path": MEDICATIONS_CSV,
        "columns": ["drug_id", "drug_name", "category", "demo_status"],
        "int": [], "float": [],
    },
    "inventory": {
        "path": INVENTORY_CSV,
        "columns": ["drug_id", "drug_name", "current_stock", "daily_usage", "reorder_level",
                    "supplier_id", "unit_price", "last_restock_date"],
        "int": ["current_stock", "reorder_level"], "float": ["daily_usage", "unit_price"],
    },
    "suppliers": {
        "path": SUPPLIERS_CSV,
        "columns": ["supplier_id", "supplier_name", "medication", "available_quantity",
                    "unit_price", "lead_time_days", "supplier_status"],
        "int": ["available_quantity", "lead_time_days"], "float": ["unit_price"],
    },
    "shortage_events": {
        "path": SHORTAGE_EVENTS_CSV,
        "columns": ["drug_id", "drug_name", "shortage_status", "reported_date", "source", "severity", "notes"],
        "int": ["severity"], "float": [],
    },
}

_ACTIVE_STATUS_MAP = {"1": 1, "true": 1, "yes": 1, "y": 1, "active": 1,
                      "0": 0, "false": 0, "no": 0, "n": 0, "inactive": 0}


def _require_file(path: Path, what: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{what} not found: {path}\n"
            f"DATASET_BASE_DIR = {DATASET_BASE_DIR}\n"
            "Make sure the Kaggle dataset 'medsupply-sentinel-demo-data' is attached to this notebook."
        )
    return path


def _normalize_active_status(series: pd.Series, where: str) -> pd.Series:
    """Accept 1/0, True/False, ACTIVE/INACTIVE ... and return 1/0 (what the workflow expects)."""
    def conv(v):
        key = str(v).strip().lower()
        if key in _ACTIVE_STATUS_MAP:
            return _ACTIVE_STATUS_MAP[key]
        try:
            return 1 if int(float(key)) == 1 else 0
        except ValueError:
            raise ValueError(f"{where}: unsupported active_status value {v!r}")
    return series.map(conv).astype("int64")


def load_local_table(name: str, path: Optional[Path] = None) -> pd.DataFrame:
    """Read one local CSV with validated columns and proper dtypes."""
    spec = _TABLE_SPECS[name]
    csv_path = _require_file(path if path is not None else spec["path"], f"Local {name} CSV")
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in spec["columns"] if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name}: missing columns {missing}; found {list(df.columns)}")
    df = df[spec["columns"]].copy()

    numeric = set(spec["int"]) | set(spec["float"])
    for col in spec["columns"]:
        if col in numeric:
            continue
        df[col] = df[col].fillna("").astype(str).str.strip()      # text: no NaN, no stray spaces

    for col in spec["int"] + spec["float"]:
        if col == "active_status":
            df[col] = _normalize_active_status(df[col], csv_path.name)
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.isna().any():
            raise ValueError(f"{csv_path.name}: column '{col}' has missing / non-numeric values")
        if col in spec["int"]:
            if (s % 1 != 0).any():
                raise ValueError(f"{csv_path.name}: column '{col}' must contain whole numbers")
            df[col] = s.astype("int64")
        else:
            df[col] = s.astype("float64")
    return df.reset_index(drop=True)


def load_demo_config() -> Dict[str, Any]:
    """Read config/demo_scenario.json from the dataset."""
    return json.loads(_require_file(CONFIG_FILE, "Demo scenario config").read_text(encoding="utf-8"))

from pypdf import PdfReader


def _to_native(record: Dict[str, Any]) -> Dict[str, Any]:
    """numpy scalars -> plain Python scalars (safe for pydantic / JSON / sqlite)."""
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in record.items()}


class LocalShortageAdapter:
    """Local replacement for an external shortage API: reads synthetic/shortage_events.csv."""

    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path = Path(csv_path) if csv_path is not None else SHORTAGE_EVENTS_CSV

    def fetch_event(self, drug_id: str) -> Dict[str, Any]:
        df = load_local_table("shortage_events", self.csv_path)
        row = df[df["drug_id"] == drug_id]
        if row.empty:
            raise KeyError(f"Medication not found in local shortage source: {drug_id}")
        return _to_native(row.iloc[0].to_dict())


class LocalInventoryAdapter:
    """Local CSV replacement for Google Sheets: reads synthetic/inventory.csv."""

    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)

    def get_inventory(self, drug_id: str) -> Dict[str, Any]:
        df = load_local_table("inventory", self.csv_path)
        row = df[df["drug_id"] == drug_id]
        if row.empty:
            raise KeyError(f"Drug {drug_id} not found in local inventory source.")
        return _to_native(row.iloc[0].to_dict())


class GuidelineRepository:
    """Reads the guideline PDFs directly from <dataset>/guidelines (no URLs, no Drive)."""

    def __init__(self, guideline_dir: Optional[Path] = None):
        self.guideline_dir = Path(guideline_dir) if guideline_dir is not None else GUIDELINE_DIR

    def list_relevant(self, drug_name: str) -> List[Dict[str, Any]]:
        if not self.guideline_dir.is_dir():
            raise FileNotFoundError(f"Guidelines directory not found: {self.guideline_dir}")
        pdfs = sorted(self.guideline_dir.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"No guideline PDFs found in: {self.guideline_dir}")
        matches = []
        for pdf in pdfs:
            reader = PdfReader(str(pdf))
            for page_num, page in enumerate(reader.pages, start=1):
                txt = page.extract_text() or ""
                if drug_name.lower() in txt.lower() or "shortage" in txt.lower():
                    matches.append({"source_name": pdf.name, "page": page_num, "text": txt})
        return matches


class SyntheticPatientRepository:
    """Active synthetic patients for a medication: reads synthetic/synthetic_patients.csv."""

    def __init__(self, csv_path: Optional[Path] = None, engine: Any = None):
        self.csv_path = Path(csv_path) if csv_path is not None else PATIENTS_CSV
        self.engine = engine          # accepted for compatibility, intentionally unused

    def impacted(self, drug_name: str) -> pd.DataFrame:
        df = load_local_table("patients", self.csv_path)
        out = df[(df["medication"] == drug_name) & (df["active_status"] == 1)]
        return out.reset_index(drop=True)


class SupplierRepository:
    """Active supplier candidates for a medication: reads synthetic/suppliers.csv."""

    def __init__(self, csv_path: Optional[Path] = None, engine: Any = None):
        self.csv_path = Path(csv_path) if csv_path is not None else SUPPLIERS_CSV
        self.engine = engine          # accepted for compatibility, intentionally unused

    def candidates(self, drug_name: str) -> pd.DataFrame:
        df = load_local_table("suppliers", self.csv_path)
        out = df[(df["medication"] == drug_name) & (df["supplier_status"] == "ACTIVE")]
        return out.sort_values("unit_price", ascending=True, kind="stable").reset_index(drop=True)


shortage_api = LocalShortageAdapter()
inventory_source = LocalInventoryAdapter(INVENTORY_CSV)
guidelines = GuidelineRepository()
patients = SyntheticPatientRepository()
suppliers = SupplierRepository()


# 4. Pydantic schemas
class Evidence(BaseModel):
    source_type: str
    source_name: str
    locator: str
    excerpt: str = ""
    fact_or_analysis: Literal["FACT", "AI_ANALYSIS"] = "FACT"

class ShortageEvent(BaseModel):
    drug_id: str
    drug_name: str
    shortage_status: Literal["NORMAL", "SHORTAGE", "CRITICAL_SHORTAGE"]
    reported_date: str
    source: str
    severity: int = Field(ge=0, le=100)
    notes: str = ""
    evidence: list[Evidence] = []

class InventoryResult(BaseModel):
    drug_name: str
    current_stock: int
    daily_usage: float
    coverage_days: float
    reorder_level: int
    sufficient_stock: bool
    urgency: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    calculation: str
    supplier_id: str
    unit_price: float
    last_restock_date: str
    evidence: list[Evidence] = []

class GuidelineResult(BaseModel):
    drug_name: str
    documents_found: list[str]
    applicable_guidance: list[str]
    constraints: list[str]
    evidence: list[Evidence] = []
    uncertainty: list[str] = []

class PatientImpactResult(BaseModel):
    affected_patient_count: int
    affected_categories: list[str]
    high_priority_cases: int
    operational_urgency: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    reasoning: str
    evidence: list[Evidence] = []

class CandidateAlternative(BaseModel):
    medication: str
    supplier_name: str
    available_quantity: int
    unit_price: float
    lead_time_days: int
    rationale: str
    constraints: list[str]
    uncertainty: str

class SubstitutionResult(BaseModel):
    candidate_alternatives: list[CandidateAlternative]
    supporting_evidence: list[Evidence]
    constraints: list[str]
    uncertainties: list[str]
    requires_pharmacist_review: bool = True

class ValidationResult(BaseModel):
    status: Literal["PENDING_REVIEW", "APPROVED", "REJECTED", "NEEDS_MORE_INFORMATION"]
    required_evidence_present: bool
    conflicts: list[str]
    reasons: list[str]
    human_review_required: bool = True

class PurchaseOrder(BaseModel):
    po_number: str
    date: str
    supplier: str
    medication: str
    quantity: int
    unit_price: float
    estimated_total: float
    priority: str
    reason: str
    status: Literal["DRAFT", "PROCUREMENT_PENDING", "CANCELLED"] = "DRAFT"
    approval_status: str = "APPROVED"

class WorkflowState(TypedDict, total=False):
    workflow_id: str
    event: dict[str, Any]
    shortage_data: dict[str, Any]
    inventory_data: dict[str, Any]
    guideline_data: dict[str, Any]
    patient_impact: dict[str, Any]
    candidate_alternatives: dict[str, Any]
    validation_result: dict[str, Any]
    approval_status: str
    reviewer_role: str
    decision_reason: str
    purchase_order: dict[str, Any]
    notifications: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[dict[str, Any]], operator.add]
    agent_logs: Annotated[list[dict[str, Any]], operator.add]
    timestamps: dict[str, str]
    execution_metrics: dict[str, Any]
    final_report: str

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def add_log(state, agent, status, started, output=None, error=None, retry_count=0):
    duration = time.perf_counter() - started
    return {
        "workflow_id": state.get("workflow_id", ""),
        "agent": agent,
        "event": f"{agent}:{status}",
        "timestamp": now_iso(),
        "status": status,
        "input_metadata": {"workflow_id": state.get("workflow_id")},
        "output_metadata": {"keys": list(output.keys()) if isinstance(output, dict) else None},
        "error": str(error) if error else None,
        "retry_count": retry_count,
        "duration_seconds": round(duration, 4),
    }

schema_sql = [
    """CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        role TEXT NOT NULL
    )""",

    """CREATE TABLE IF NOT EXISTS medications (
        drug_id TEXT PRIMARY KEY,
        drug_name TEXT UNIQUE NOT NULL
    )""",

    """CREATE TABLE IF NOT EXISTS inventory (
        drug_id TEXT PRIMARY KEY,
        drug_name TEXT NOT NULL,
        current_stock INTEGER NOT NULL,
        daily_usage REAL NOT NULL,
        reorder_level INTEGER NOT NULL,
        supplier_id TEXT,
        unit_price REAL,
        last_restock_date TEXT
    )""",

    """CREATE TABLE IF NOT EXISTS suppliers (
        supplier_id TEXT,
        supplier_name TEXT,
        medication TEXT,
        available_quantity INTEGER,
        unit_price REAL,
        lead_time_days INTEGER,
        supplier_status TEXT
    )""",

    """CREATE TABLE IF NOT EXISTS synthetic_patients (
        patient_id TEXT PRIMARY KEY,
        age_group TEXT,
        condition_category TEXT,
        medication TEXT,
        risk_group TEXT,
        active_status INTEGER
    )""",

    """CREATE TABLE IF NOT EXISTS shortage_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        drug_id TEXT,
        drug_name TEXT,
        shortage_status TEXT,
        reported_date TEXT,
        source TEXT,
        severity INTEGER,
        notes TEXT
    )""",

    """CREATE TABLE IF NOT EXISTS workflow_runs (
        workflow_id TEXT PRIMARY KEY,
        status TEXT,
        created_at TEXT,
        updated_at TEXT
    )""",

    """CREATE TABLE IF NOT EXISTS agent_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workflow_id TEXT,
        agent TEXT,
        event TEXT,
        timestamp TEXT,
        status TEXT,
        input_metadata TEXT,
        output_metadata TEXT,
        error TEXT,
        retry_count INTEGER,
        duration_seconds REAL
    )""",

    """CREATE TABLE IF NOT EXISTS validation_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workflow_id TEXT,
        status TEXT,
        reasons TEXT,
        decision_reason TEXT,
        reviewer_role TEXT,
        timestamp TEXT
    )""",

    """CREATE TABLE IF NOT EXISTS purchase_orders (
        po_number TEXT PRIMARY KEY,
        workflow_id TEXT,
        date TEXT,
        supplier TEXT,
        medication TEXT,
        quantity INTEGER,
        unit_price REAL,
        estimated_total REAL,
        priority TEXT,
        reason TEXT,
        status TEXT,
        approval_status TEXT
    )""",

    """CREATE TABLE IF NOT EXISTS notifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workflow_id TEXT,
        channel TEXT,
        status TEXT,
        recipient TEXT,
        subject TEXT,
        timestamp TEXT,
        error TEXT
    )"""
]



engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


def _py(v):
    return v.item() if hasattr(v, "item") else v


def _rows(df):
    return [tuple(_py(v) for v in row) for row in df.itertuples(index=False, name=None)]


def init_db() -> None:
    """Create tables if missing and (re)seed the reference tables from the CSV dataset.
    Audit tables (workflow_runs, agent_logs, validation_results, purchase_orders, notifications) are kept."""
    medications_df = load_local_table("medications")
    inventory_df = load_local_table("inventory")
    suppliers_df = load_local_table("suppliers")
    shortage_events_df = load_local_table("shortage_events")
    patients_df = load_local_table("patients")
    with engine.begin() as conn:
        for stmt in schema_sql:
            conn.exec_driver_sql(stmt)
        for t in ("users", "medications", "inventory", "suppliers", "shortage_events", "synthetic_patients"):
            conn.exec_driver_sql(f"DELETE FROM {t}")
        conn.exec_driver_sql("INSERT INTO users (user_id, role) VALUES (?, ?)", (1, "PHARMACIST"))
        for r in _rows(medications_df[["drug_id", "drug_name"]]):
            conn.exec_driver_sql("INSERT INTO medications (drug_id, drug_name) VALUES (?, ?)", r)
        for r in _rows(inventory_df):
            conn.exec_driver_sql("INSERT INTO inventory VALUES (?, ?, ?, ?, ?, ?, ?, ?)", r)
        for r in _rows(suppliers_df):
            conn.exec_driver_sql("INSERT INTO suppliers VALUES (?, ?, ?, ?, ?, ?, ?)", r)
        for r in _rows(shortage_events_df):
            conn.exec_driver_sql(
                "INSERT INTO shortage_events (drug_id,drug_name,shortage_status,reported_date,source,severity,notes)"
                " VALUES (?,?,?,?,?,?,?)", r)
        for r in _rows(patients_df):
            conn.exec_driver_sql("INSERT INTO synthetic_patients VALUES (?, ?, ?, ?, ?, ?)", r)


def list_medications() -> pd.DataFrame:
    return load_local_table("medications")[["drug_id", "drug_name"]]

AGENT_PROMPTS = {

    "shortage_monitor": """
You are the Shortage Monitor Agent.

Extract only facts explicitly present in the supplied shortage data.

Rules:
- Do not invent dates.
- Do not invent shortage statuses.
- Do not invent suppliers.
- Do not invent quantities.
- Do not invent clinical claims.
- If information is missing, explicitly say it is missing.
""",

    "inventory_analyst": """
You are the Inventory Analyst Agent.

Use only the supplied inventory facts.

Calculate:

coverage_days = current_stock / daily_usage

Explain the calculation clearly.

Classify operational urgency based only on the supplied
inventory and shortage information.

Rules:
- Do not invent inventory values.
- Do not invent demand.
- Do not make treatment decisions.
- Do not prescribe medication.
- If required information is missing, state that clearly.
""",

    "guideline_reader": """
You are the Guideline Reader Agent.

Use only the retrieved PDF text.

Summarize applicable operational guidance.

Rules:
- Preserve document/page evidence when provided.
- Do not fabricate guideline content.
- Do not create recommendations that are not supported by the
  retrieved document.
- If evidence is insufficient, explicitly state that.
""",

    "patient_impact": """
You are the Patient Impact Analyst.

Use synthetic patient records only.

Report:
- operational impact
- patient count
- relevant categories
- urgency

Rules:
- Do not diagnose patients.
- Do not prescribe.
- Do not invent patient information.
- Do not infer medical conditions that are not explicitly supplied.
""",

    "substitution_analyst": """
You are the Substitution Analyst.

Compare only:
1. approved candidate product/supplier data
2. retrieved guideline constraints

Generate candidate alternatives for pharmacist review only.

Rules:
- Never prescribe.
- Never switch therapy autonomously.
- Never specify dosage.
- Never invent contraindications.
- Never invent product information.
- Clearly identify uncertainty.
""",

    "validator": """
You are the Pharmacist Validator Agent.

Check:
- evidence completeness
- conflicting information
- missing information
- uncertainty
- human-review requirements

You are a validation gate, not an autonomous clinical authority.

Rules:
- Do not invent evidence.
- Do not make unsupported clinical decisions.
- Clearly identify when human review is required.
""",

    "final_report": """
You are the Final Reporting Agent.

Produce a concise operational summary.

Separate the output into:

FACTS
EVIDENCE
AI-GENERATED ANALYSIS
HUMAN APPROVAL
OPERATIONAL ACTIONS

Rules:
- Do not introduce unsupported facts.
- Do not invent evidence.
- Do not make autonomous clinical decisions.
- Clearly distinguish AI analysis from human approval.
"""
}



# ===============================================================
# LLM layer — Groq API (replaces local Mistral-Nemo)
# Same interface as the notebook: llm_client.summarize(agent_name, payload) -> str
# ===============================================================
GROQ_DEFAULT_MODEL = "openai/gpt-oss-20b"
GROQ_FALLBACK_MODEL = "llama-3.3-70b-versatile"
MAX_INPUT_CHARS = 14000
MAX_TOKENS = 400
TEMPERATURE = 0.2


def _build_messages(agent_name: str, payload: dict) -> list:
    if agent_name not in AGENT_PROMPTS:
        raise ValueError(f"Unknown agent: {agent_name}. Available: {list(AGENT_PROMPTS)}")
    payload_json = json.dumps(payload, default=str, ensure_ascii=False, indent=2)
    if len(payload_json) > MAX_INPUT_CHARS:
        payload_json = payload_json[:MAX_INPUT_CHARS] + "\n[INPUT TRUNCATED]"
    return [
        {"role": "system", "content": AGENT_PROMPTS[agent_name]},
        {"role": "user", "content": "Analyze ONLY the grounded workflow data below.\n"
                                    "Do not add facts.\n\nWORKFLOW DATA:\n" + payload_json},
    ]


class GroqLLM:
    """Groq chat-completions client. If the primary model fails it tries a fallback model;
    if both fail it returns a marker string so the deterministic workflow still completes
    (the LLM text is explanatory only — validation and human approval stay authoritative)."""

    def __init__(self, api_key: str, model: str = GROQ_DEFAULT_MODEL, fallback_model: str = GROQ_FALLBACK_MODEL):
        from groq import Groq
        self.client = Groq(api_key=api_key, max_retries=2, timeout=30.0)
        self.model = model
        self.fallback_model = fallback_model
        self.last_model_used: Optional[str] = None
        self.last_error: Optional[str] = None

    def summarize(self, agent_name: str, payload: dict) -> str:
        messages = _build_messages(agent_name, payload)
        models = [self.model] + ([self.fallback_model] if self.fallback_model and self.fallback_model != self.model else [])
        for m in models:
            try:
                resp = self.client.chat.completions.create(
                    model=m, messages=messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, top_p=0.9)
                self.last_model_used, self.last_error = m, None
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:  # rate limit, network, bad model id ...
                self.last_error = f"{type(exc).__name__}: {exc}"
        return f"[LLM unavailable — deterministic result only. {self.last_error}]"


class OfflineLLM:
    """No API call — lets the app run (and be tested) without a Groq key."""
    model = "offline"
    last_model_used = "offline"
    last_error = None

    def summarize(self, agent_name: str, payload: dict) -> str:
        _build_messages(agent_name, payload)   # still validates the agent name
        return f"[offline mode — no LLM call] {agent_name}: see the deterministic fields."


llm_client = OfflineLLM()


def set_llm(api_key: Optional[str], model: str = GROQ_DEFAULT_MODEL) -> str:
    """Select the LLM used by all agents. Returns a short status string."""
    global llm_client
    if api_key:
        llm_client = GroqLLM(api_key=api_key, model=model)
        return f"Groq · {model}"
    llm_client = OfflineLLM()
    return "Offline (no API key)"

# 10. Specialized agents

def sentinel_orchestrator(state: WorkflowState) -> dict:
    started = time.perf_counter()
    workflow_id = state["workflow_id"]
    with engine.begin() as conn:
        conn.exec_driver_sql(
            """INSERT OR REPLACE INTO workflow_runs
               (workflow_id, status, created_at, updated_at) VALUES (?, ?, ?, ?)""",
            (workflow_id, "RUNNING", now_iso(), now_iso())
        )
    log = add_log(state, "Sentinel Orchestrator", "COMPLETED", started, {"workflow_id": workflow_id})
    return {"timestamps": {"orchestrator_start": now_iso()}, "agent_logs": [log]}

def shortage_monitor_agent(state: WorkflowState) -> dict:
    started = time.perf_counter()
    event = shortage_api.fetch_event(state["event"]["drug_id"])
    evidence = Evidence(
        source_type="API",
        source_name=event["source"],
        locator=f"drug_id={event['drug_id']}",
        excerpt=f"{event['drug_name']} status={event['shortage_status']} severity={event['severity']}",
        fact_or_analysis="FACT"
    )
    result = ShortageEvent(**event, evidence=[evidence])
    log = add_log(state, "Shortage Monitor Agent", "COMPLETED", started, result.model_dump())
    return {"shortage_data": result.model_dump(), "agent_logs": [log]}

def inventory_agent(state: WorkflowState) -> dict:
    started = time.perf_counter()
    d = state["shortage_data"]
    row = inventory_source.get_inventory(d["drug_id"])
    coverage = float(row["current_stock"]) / max(float(row["daily_usage"]), 1e-9)
    urgency = (
        "CRITICAL" if coverage < 2 else
        "HIGH" if coverage < 4 or row["current_stock"] <= row["reorder_level"] else
        "MEDIUM" if coverage < 7 else "LOW"
    )
    result = InventoryResult(
        drug_name=row["drug_name"],
        current_stock=int(row["current_stock"]),
        daily_usage=float(row["daily_usage"]),
        coverage_days=round(coverage, 2),
        reorder_level=int(row["reorder_level"]),
        sufficient_stock=bool(coverage >= 7),
        urgency=urgency,
        calculation=f"coverage_days = {int(row['current_stock'])} / {float(row['daily_usage'])} = {coverage:.2f} days",
        supplier_id=row["supplier_id"],
        unit_price=float(row["unit_price"]),
        last_restock_date=str(row["last_restock_date"]),
        evidence=[Evidence(
            source_type="INVENTORY_SHEET",
            source_name="inventory.csv (Google Sheets-compatible demo)",
            locator=f"drug_id={row['drug_id']}",
            excerpt=f"stock={row['current_stock']}, daily_usage={row['daily_usage']}, reorder_level={row['reorder_level']}",
            fact_or_analysis="FACT"
        )]
    )
    log = add_log(state, "Inventory Analyst Agent", "COMPLETED", started, result.model_dump())
    return {"inventory_data": result.model_dump(), "agent_logs": [log]}

def guideline_agent(state: WorkflowState) -> dict:
    started = time.perf_counter()
    drug = state["shortage_data"]["drug_name"]
    docs = guidelines.list_relevant(drug)
    if not docs:
        raise RuntimeError(f"No guideline evidence found for {drug}")
    applicable, constraints, evidence = [], [], []
    for doc in docs:
        txt = re.sub(r"\s+", " ", doc["text"]).strip()
        if "shortage" in txt.lower() or "supply" in txt.lower():
            applicable.append(txt[:400])
        if "pharmacist review" in txt.lower():
            constraints.append("Candidate alternatives require qualified pharmacist review.")
        evidence.append(Evidence(
            source_type="GUIDELINE_PDF",
            source_name=doc["source_name"],
            locator=f"page={doc['page']}",
            excerpt=txt[:500],
            fact_or_analysis="FACT"
        ))
    llm_summary = llm_client.summarize("guideline_reader", {"drug": drug, "documents": docs})
    result = GuidelineResult(
        drug_name=drug,
        documents_found=sorted({d["source_name"] for d in docs}),
        applicable_guidance=applicable,
        constraints=sorted(set(constraints)),
        evidence=evidence
    )
    log = add_log(state, "Guideline Reader Agent", "COMPLETED", started, result.model_dump())
    return {"guideline_data": result.model_dump(), "agent_logs": [log]}

def patient_impact_agent(state: WorkflowState) -> dict:
    started = time.perf_counter()
    drug = state["shortage_data"]["drug_name"]
    df = patients.impacted(drug)
    high = int((df["risk_group"] == "HIGH").sum())
    categories = sorted(df["condition_category"].value_counts().index.tolist())
    urgency = (
        "CRITICAL" if high >= 5 else
        "HIGH" if len(df) >= 10 else
        "MEDIUM" if len(df) > 0 else "LOW"
    )
    llm_summary = llm_client.summarize("patient_impact", {"drug": drug, "patient_count": len(df), "high_priority": high, "categories": categories})
    result = PatientImpactResult(
        affected_patient_count=int(len(df)),
        affected_categories=categories,
        high_priority_cases=high,
        operational_urgency=urgency,
        reasoning=llm_summary,
        evidence=[Evidence(
            source_type="SYNTHETIC_DB",
            source_name="synthetic_patients",
            locator=f"SQL: medication='{drug}' AND active_status=1",
            excerpt=f"{len(df)} active synthetic records matched.",
            fact_or_analysis="FACT"
        )]
    )
    log = add_log(state, "Patient Impact Analyst Agent", "COMPLETED", started, result.model_dump())
    return {"patient_impact": result.model_dump(), "agent_logs": [log]}

def substitution_agent(state: WorkflowState) -> dict:
    started = time.perf_counter()
    drug = state["shortage_data"]["drug_name"]
    cands = suppliers.candidates(drug)
    if cands.empty:
        result = SubstitutionResult(
            candidate_alternatives=[],
            supporting_evidence=[],
            constraints=["No suitable candidate records available."],
            uncertainties=["Supplier/product source returned no eligible records."],
            requires_pharmacist_review=True,
        )
    else:
        alts = []
        for _, r in cands.head(3).iterrows():
            alts.append(CandidateAlternative(
                medication=drug,
                supplier_name=str(r["supplier_name"]),
                available_quantity=int(r["available_quantity"]),
                unit_price=float(r["unit_price"]),
                lead_time_days=int(r["lead_time_days"]),
                rationale="Candidate supplier option based on availability, price, and lead time; this is not a therapeutic substitution decision.",
                constraints=state["guideline_data"].get("constraints", []),
                uncertainty="Pharmacist must validate appropriateness before any clinical use."
            ))
        evidence = [Evidence(
            source_type="SUPPLIER_DB",
            source_name="suppliers",
            locator=f"medication='{drug}'",
            excerpt=f"{len(cands)} active supplier records found.",
            fact_or_analysis="FACT"
        )]
        result = SubstitutionResult(
            candidate_alternatives=alts,
            supporting_evidence=evidence,
            constraints=state["guideline_data"].get("constraints", []),
            uncertainties=["Clinical appropriateness is intentionally outside this system's autonomous scope."],
            requires_pharmacist_review=True,
        )
    llm_client.summarize("substitution_analyst", {"drug": drug, "candidates": result.model_dump()})
    log = add_log(state, "Substitution Analyst Agent", "COMPLETED", started, result.model_dump())
    return {"candidate_alternatives": result.model_dump(), "agent_logs": [log]}

def validator_agent(state: WorkflowState) -> dict:
    started = time.perf_counter()
    reasons, conflicts = [], []
    evidence_ok = all([
        bool(state.get("shortage_data")),
        bool(state.get("inventory_data")),
        bool(state.get("guideline_data")),
        bool(state.get("patient_impact"))
    ])
    if not evidence_ok:
        reasons.append("One or more required evidence bundles are missing.")
    if not state.get("candidate_alternatives", {}).get("requires_pharmacist_review", True):
        reasons.append("Safety invariant violated: pharmacist review flag must remain true.")
    if not state.get("candidate_alternatives", {}).get("candidate_alternatives"):
        reasons.append("No candidate supplier/product records are available.")
    else:
        reasons.append("Evidence is complete enough for pharmacist review; human approval is mandatory before operational procurement action.")
    result = ValidationResult(
        status="PENDING_REVIEW" if evidence_ok else "NEEDS_MORE_INFORMATION",
        required_evidence_present=evidence_ok,
        conflicts=conflicts,
        reasons=reasons,
        human_review_required=True,
    )
    with engine.begin() as conn:
        conn.exec_driver_sql(
            """INSERT INTO validation_results
               (workflow_id,status,reasons,decision_reason,reviewer_role,timestamp)
               VALUES (?,?,?,?,?,?)""",
            (state["workflow_id"], result.status, json.dumps(result.reasons), "", "", now_iso())
        )
    log = add_log(state, "Pharmacist Validator Agent", "COMPLETED", started, result.model_dump())
    return {"validation_result": result.model_dump(), "approval_status": result.status, "agent_logs": [log]}

def pharmacist_review_gate(state: WorkflowState) -> dict:
    started = time.perf_counter()
    payload = {
        "message": "PHARMACIST REVIEW REQUIRED",
        "workflow_id": state["workflow_id"],
        "drug": state["shortage_data"]["drug_name"],
        "shortage_status": state["shortage_data"]["shortage_status"],
        "coverage_days": state["inventory_data"]["coverage_days"],
        "affected_synthetic_patients": state["patient_impact"]["affected_patient_count"],
        "candidate_count": len(state["candidate_alternatives"].get("candidate_alternatives", [])),
        "options": ["APPROVED", "REJECTED", "NEEDS_MORE_INFORMATION"],
        "safety_note": "Candidate alternatives are operational review items only."
    }
    decision = interrupt(payload)
    status = decision.get("status", "REJECTED") if isinstance(decision, dict) else str(decision)
    role = decision.get("reviewer_role", "PHARMACIST") if isinstance(decision, dict) else "PHARMACIST"
    reason = decision.get("reason", "") if isinstance(decision, dict) else ""
    log = add_log(state, "Pharmacist Human Gate", "COMPLETED", started, {"status": status})
    return {
        "approval_status": status,
        "reviewer_role": role,
        "decision_reason": reason,
        "agent_logs": [log]
    }

def purchase_order_agent(state: WorkflowState) -> dict:
    started = time.perf_counter()
    drug = state["shortage_data"]["drug_name"]
    inv = state["inventory_data"]
    cands = state["candidate_alternatives"]["candidate_alternatives"]
    chosen = cands[0]
    target_days = max(7, inv["reorder_level"])
    quantity = max(1, math.ceil(target_days * inv["daily_usage"] - inv["current_stock"]))
    quantity = min(quantity, int(chosen["available_quantity"]))
    po = PurchaseOrder(
        po_number=f"MS-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        supplier=chosen["supplier_name"],
        medication=drug,
        quantity=quantity,
        unit_price=float(chosen["unit_price"]),
        estimated_total=round(quantity * float(chosen["unit_price"]), 2),
        priority=inv["urgency"],
        reason=f"Shortage status={state['shortage_data']['shortage_status']} with {inv['coverage_days']:.2f} days projected coverage.",
        status="DRAFT",
        approval_status="APPROVED",
    )
    log = add_log(state, "Purchase Order Agent", "COMPLETED", started, po.model_dump())
    with engine.begin() as conn:
        conn.exec_driver_sql(
            """INSERT INTO purchase_orders
               (po_number,workflow_id,date,supplier,medication,quantity,unit_price,
                estimated_total,priority,reason,status,approval_status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                po.po_number, state["workflow_id"], po.date, po.supplier, po.medication,
                po.quantity, po.unit_price, po.estimated_total, po.priority, po.reason,
                po.status, po.approval_status
            )
        )
    return {"purchase_order": po.model_dump(), "agent_logs": [log]}

def notification_agent(state: WorkflowState) -> dict:
    started = time.perf_counter()
    drug = state["shortage_data"]["drug_name"]
    po = state["purchase_order"]
    subject = f"Medication Shortage Alert — {drug}"
    body = (
        f"Shortage: {state['shortage_data']['shortage_status']}\n"
        f"Inventory: {state['inventory_data']['current_stock']} units\n"
        f"Coverage: {state['inventory_data']['coverage_days']} days\n"
        f"Synthetic patient impact: {state['patient_impact']['affected_patient_count']}\n"
        f"Validation: {state['validation_result']['status']}\n"
        f"Human approval: {state['approval_status']}\n"
        f"PO: {po['po_number']} / {po['estimated_total']:.2f}\n"
    )
    result = {
        "channel": "EMAIL",
        "status": "SIMULATED_SENT",
        "recipient": "pharmacy-demo@example.org",
        "subject": subject,
        "body": body
    }
    with engine.begin() as conn:
        conn.exec_driver_sql(
            """INSERT INTO notifications
               (workflow_id,channel,status,recipient,subject,timestamp,error)
               VALUES (?,?,?,?,?,?,?)""",
            (state["workflow_id"], "EMAIL", result["status"], result["recipient"], subject, now_iso(), None)
        )
    log = add_log(state, "Notification Agent", "COMPLETED", started, {"status": result["status"]})
    return {"notifications": [result], "agent_logs": [log]}

def final_report_agent(state: WorkflowState) -> dict:
    started = time.perf_counter()
    sd = state["shortage_data"]
    inv = state["inventory_data"]
    gd = state["guideline_data"]
    pi = state["patient_impact"]
    sub = state["candidate_alternatives"]
    po = state.get("purchase_order")

    report = f"""# MedSupply Sentinel — Incident Report

## FACTS
- Drug: {sd['drug_name']}
- Shortage status: {sd['shortage_status']}
- Reported date: {sd['reported_date']}
- Source: {sd['source']}
- Severity score: {sd['severity']}
- Current stock: {inv['current_stock']} units
- Daily usage: {inv['daily_usage']:.2f} units/day
- Estimated coverage: {inv['coverage_days']:.2f} days
- Active synthetic patients impacted: {pi['affected_patient_count']}
- High-priority synthetic cases: {pi['high_priority_cases']}

## EVIDENCE
- Shortage source: {sd['evidence'][0]['locator']}
- Inventory source: {inv['evidence'][0]['locator']}
- Guideline documents: {', '.join(gd['documents_found'])}
- Patient database query: {pi['evidence'][0]['locator']}

## AI-GENERATED ANALYSIS
- Inventory calculation: {inv['calculation']}
- Operational urgency: {inv['urgency']}
- Patient-impact reasoning: {pi['reasoning']}
- Candidate options generated: {len(sub.get('candidate_alternatives', []))}
- Uncertainty: {'; '.join(sub.get('uncertainties', []))}

## HUMAN APPROVAL
- Status: {state['approval_status']}
- Reviewer role: {state.get('reviewer_role', 'N/A')}
- Decision reason: {state.get('decision_reason', '') or 'N/A'}

## OPERATIONAL ACTIONS
"""
    if po:
        report += f"- PO draft: {po['po_number']} — {po['quantity']} units — estimated total {po['estimated_total']:.2f}\n"
    else:
        report += "- No procurement action generated.\n"
    report += f"- Notifications: {len(state.get('notifications', []))}\n"
    report += "\n## SAFETY NOTE\nThis report is operational decision support only. Candidate alternatives are not treatment prescriptions and require qualified pharmacist review.\n"
    log = add_log(state, "Final Reporting Agent", "COMPLETED", started, {"report_chars": len(report)})
    return {"final_report": report, "agent_logs": [log]}

print("Specialized agents defined.")

def order_quantity_plan(state: WorkflowState) -> dict[str, Any]:
    """Compatibility helper used by the Streamlit pharmacist-review screen.

    Returns a dict with the shape expected by app.py:
      - quantity: recommended order amount
      - supplier: chosen supplier name or None
      - reason: short justification
      - max_quantity: highest allowed order amount from the chosen supplier
    """
    cands = state.get("candidate_alternatives", {}).get("candidate_alternatives", [])
    inv = state.get("inventory_data") or {}

    if not cands:
        return {
            "quantity": 0,
            "supplier": None,
            "reason": "No eligible supplier alternatives are available for an order.",
            "max_quantity": 0,
        }

    chosen = min(
        cands,
        key=lambda c: (
            float(c.get("unit_price", float("inf"))),
            -int(c.get("available_quantity", 0)),
        ),
    )

    stock = int(inv.get("current_stock", 0))
    daily_usage = float(inv.get("daily_usage", 0.0))
    target_days = max(7, int(inv.get("reorder_level", 7)))
    required = max(0, math.ceil(target_days * daily_usage - stock))
    available = int(chosen.get("available_quantity", 0))
    quantity = min(required, available) if available > 0 else 0
    if quantity <= 0:
        return {
            "quantity": 0,
            "supplier": chosen.get("supplier_name"),
            "reason": (
                f"Supplier {chosen.get('supplier_name')} has no available units to cover the shortage."
            ),
            "max_quantity": 0,
        }

    reason = (
        f"Target coverage is {target_days} days; current stock {stock} units and daily usage "
        f"{daily_usage:.2f} imply a refill of {required} units, capped by supplier availability."
    )

    return {
        "quantity": int(quantity),
        "supplier": chosen.get("supplier_name"),
        "reason": reason,
        "max_quantity": int(available),
    }


def route_after_approval(state: WorkflowState):
    if state.get("approval_status") == "APPROVED":
        return "purchase_order"
    return "final_report"

def build_graph():
    """Compile the LangGraph workflow with an in-memory checkpointer (pause/resume at the pharmacist gate)."""
    builder = StateGraph(WorkflowState)

    builder.add_node("sentinel", sentinel_orchestrator)
    builder.add_node("shortage_monitor", shortage_monitor_agent)
    builder.add_node("inventory", inventory_agent)
    builder.add_node("guideline", guideline_agent)
    builder.add_node("patient_impact", patient_impact_agent)
    builder.add_node("substitution", substitution_agent)
    builder.add_node("validator", validator_agent)
    builder.add_node("pharmacist_review", pharmacist_review_gate)
    builder.add_node("purchase_order", purchase_order_agent)
    builder.add_node("notification", notification_agent)
    builder.add_node("final_report", final_report_agent)

    builder.add_edge(START, "sentinel")
    builder.add_edge("sentinel", "shortage_monitor")

    builder.add_edge("shortage_monitor", "inventory")
    builder.add_edge("shortage_monitor", "guideline")

    builder.add_edge("inventory", "patient_impact")
    builder.add_edge("guideline", "patient_impact")

    builder.add_edge("patient_impact", "substitution")
    builder.add_edge("substitution", "validator")
    builder.add_edge("validator", "pharmacist_review")

    builder.add_conditional_edges(
        "pharmacist_review",
        route_after_approval,
        {"purchase_order": "purchase_order", "final_report": "final_report"}
    )

    builder.add_edge("purchase_order", "notification")
    builder.add_edge("notification", "final_report")
    builder.add_edge("final_report", END)

    return builder.compile(checkpointer=InMemorySaver())



def make_po_pdf(po: dict) -> Path:
    po_path = REPORT_DIR / f"{po['po_number']}.pdf"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CenterTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22))

    doc = SimpleDocTemplate(
        str(po_path), pagesize=A4,
        rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42
    )

    story = [
        Paragraph("MEDSUPPLY SENTINEL", styles["CenterTitle"]),
        Paragraph("PURCHASE ORDER — GENERATED DRAFT", styles["Heading2"]),
        Spacer(1, 12)
    ]

    po_table = Table([
        ["PO Number", po["po_number"], "Date", po["date"]],
        ["Supplier", po["supplier"], "Priority", po["priority"]],
        ["Medication", po["medication"], "Quantity", str(po["quantity"])],
        ["Unit Price", f"{po['unit_price']:.2f}", "Estimated Total", f"{po['estimated_total']:.2f}"],
        ["Status", po["status"], "Approval", po["approval_status"]],
        ["Reason", Paragraph(po["reason"], styles["BodyText"]), "", ""],
    ], colWidths=[90, 185, 80, 120])

    po_table.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.5,colors.grey),
        ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("SPAN",(1,5),(-1,5)),
        ("PADDING",(0,0),(-1,-1),6),
    ]))

    story += [
        po_table,
        Spacer(1, 18),
        Paragraph(
            "Safety / scope notice: This document is an operational procurement draft and does not authorize autonomous clinical medication substitution.",
            styles["BodyText"]
        )
    ]

    doc.build(story)

    return po_path


# ===============================================================
# Run helpers used by the Streamlit app
# ===============================================================
def new_workflow_state(drug_id: str) -> tuple[WorkflowState, dict]:
    workflow_id = f"WF-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
    state: WorkflowState = {
        "workflow_id": workflow_id,
        "event": {"drug_id": drug_id},
        "errors": [], "agent_logs": [], "notifications": [],
        "timestamps": {"created_at": now_iso()},
        "execution_metrics": {"retries": 0},
    }
    return state, {"configurable": {"thread_id": workflow_id}}


def persist_logs(final_state: dict) -> None:
    logs = final_state.get("agent_logs", [])
    with engine.begin() as conn:
        for r in logs:
            conn.exec_driver_sql(
                """INSERT INTO agent_logs (workflow_id,agent,event,timestamp,status,input_metadata,
                   output_metadata,error,retry_count,duration_seconds) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (r.get("workflow_id"), r.get("agent"), r.get("event"), r.get("timestamp"), r.get("status"),
                 json.dumps(r.get("input_metadata", {})), json.dumps(r.get("output_metadata", {})),
                 r.get("error"), int(r.get("retry_count", 0)), float(r.get("duration_seconds", 0))))


def finalize_run(workflow_id: str, final_state: dict) -> None:
    persist_logs(final_state)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE workflow_runs SET status=?, updated_at=? WHERE workflow_id=?",
            ("COMPLETED" if final_state.get("final_report") else "FAILED", now_iso(), workflow_id))
        conn.exec_driver_sql(
            """UPDATE validation_results SET status=?, decision_reason=?, reviewer_role=?, timestamp=?
               WHERE workflow_id=? AND id=(SELECT MAX(id) FROM validation_results WHERE workflow_id=?)""",
            (final_state.get("approval_status"), final_state.get("decision_reason", ""),
             final_state.get("reviewer_role", ""), now_iso(), workflow_id, workflow_id))


def save_report(workflow_id: str, report_md: str) -> Path:
    p = REPORT_DIR / f"{workflow_id}_incident_report.md"
    p.write_text(report_md, encoding="utf-8")
    return p


def recent_runs(limit: int = 20) -> pd.DataFrame:
    return pd.read_sql_query(
        sql_text("SELECT workflow_id,status,created_at,updated_at FROM workflow_runs ORDER BY created_at DESC LIMIT :n"),
        engine, params={"n": limit})

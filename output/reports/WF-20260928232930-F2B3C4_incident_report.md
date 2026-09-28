# MedSupply Sentinel — Incident Report

## FACTS
- Drug: Vancomycin
- Shortage status: NORMAL
- Reported date: 2026-09-28
- Source: Mock Drug Shortage API
- Severity score: 5
- Current stock: 75 units
- Daily usage: 10.00 units/day
- Estimated coverage: 7.50 days
- Active synthetic patients impacted: 15
- High-priority synthetic cases: 0

## EVIDENCE
- Shortage source: drug_id=M002
- Inventory source: drug_id=M002
- Guideline documents: antimicrobial_shortage_operations_demo.pdf, ceftriaxone_supply_continuity_demo.pdf
- Patient database query: SQL: medication='Vancomycin' AND active_status=1

## AI-GENERATED ANALYSIS
- Inventory calculation: coverage_days = 75 / 10.0 = 7.50 days
- Operational urgency: LOW
- Patient-impact reasoning: [offline mode — no LLM call] patient_impact: see the deterministic fields.
- Candidate options generated: 1
- Uncertainty: Clinical appropriateness is intentionally outside this system's autonomous scope.

## HUMAN APPROVAL
- Status: REJECTED
- Reviewer role: PHARMACIST
- Decision reason: fuck

## OPERATIONAL ACTIONS
- No procurement action generated.
- Notifications: 0

## SAFETY NOTE
This report is operational decision support only. Candidate alternatives are not treatment prescriptions and require qualified pharmacist review.

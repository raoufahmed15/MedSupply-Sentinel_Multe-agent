# MedSupply Sentinel — Incident Report

## FACTS
- Drug: Amoxicillin
- Shortage status: NORMAL
- Reported date: 2026-09-28
- Source: Mock Drug Shortage API
- Severity score: 2
- Current stock: 40 units
- Daily usage: 20.00 units/day
- Estimated coverage: 2.00 days
- Active synthetic patients impacted: 16
- High-priority synthetic cases: 1

## EVIDENCE
- Shortage source: drug_id=M004
- Inventory source: drug_id=M004
- Guideline documents: antimicrobial_shortage_operations_demo.pdf, ceftriaxone_supply_continuity_demo.pdf, uploaded_rag_update_guideline.pdf, uploaded_sample_guideline_amoxicillin.pdf
- Patient database query: SQL: medication='Amoxicillin' AND active_status=1

## AI-GENERATED ANALYSIS
- Inventory calculation: coverage_days = 40 / 20.0 = 2.00 days
- Operational urgency: HIGH
- Patient-impact reasoning: **Patient Impact Report – Amoxicillin Workflow**

- **Operational Impact**: 16 patients are scheduled to receive Amoxicillin.  
- **Patient Count**: 16  
- **Relevant Categories**: INFECTION  
- **Urgency**: High priority (high_priority = 1)
- Candidate options generated: 3
- Uncertainty: Clinical appropriateness is intentionally outside this system's autonomous scope.

## HUMAN APPROVAL
- Status: APPROVED
- Reviewer role: PHARMACIST
- Decision reason: as

## OPERATIONAL ACTIONS
- PO draft: MS-20260929-CCE05E — 1000 units — estimated total 1050.00
- Notifications: 1

## SAFETY NOTE
This report is operational decision support only. Candidate alternatives are not treatment prescriptions and require qualified pharmacist review.

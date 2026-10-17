import json
from typing import Dict, Any, List

def verify_dispatch_plan(plan: Any, state_json: str) -> Dict[str, Any]:
    """
    Verifies a proposed dispatch plan against the physical life-support invariants
    of the emergency state telemetry.

    Invariants checked:
    1. Every patient is assigned to an existing, valid ambulance and hospital.
    2. Every patient in the emergency state is assigned exactly once.
    3. No ambulance is double-booked across multiple patients.
    4. Hospital ICU bed capacity is never exceeded.
    5. Hospital ventilator capacity is never exceeded for critical patients.
    6. Transit ETA never breaches the patient's Golden-Hour survival window.
    """
    if not isinstance(plan, list):
        return {
            "valid": False,
            "violations": ["FormatError: Dispatch plan must be a list of dictionaries."],
            "sla_score": 0.0,
            "stats": {"assigned_count": 0, "total_patients": 0, "breached_count": 0}
        }

    try:
        state = json.loads(state_json) if isinstance(state_json, str) else state_json
    except Exception as e:
        return {
            "valid": False,
            "violations": [f"StateParseError: Invalid JSON state: {e}"],
            "sla_score": 0.0,
            "stats": {"assigned_count": 0, "total_patients": 0, "breached_count": 0}
        }

    patients = {p["patient_id"]: p for p in state.get("patients", [])}
    hospitals = {h_id: dict(cap) for h_id, cap in state.get("hospitals", {}).items()}
    ambulances = {a["ambulance_id"]: a["eta_matrix_mins"] for a in state.get("ambulances", [])}

    used_ambulances = set()
    assigned_patients = set()
    violations: List[str] = []
    sla_margins: List[float] = []

    breached_count = 0
    icu_overflow_count = 0
    vent_overflow_count = 0

    for idx, item in enumerate(plan):
        if not isinstance(item, dict):
            violations.append(f"FormatError: Entry at index {idx} is not a dictionary.")
            continue

        p_id = item.get("patient_id")
        a_id = item.get("ambulance_id")
        h_id = item.get("hospital_id")

        # Unwrap nested dict representations if emitted by LLM
        if isinstance(p_id, dict):
            p_id = p_id.get("patient_id") or p_id.get("id")
        if isinstance(a_id, dict):
            a_id = a_id.get("ambulance_id") or a_id.get("id")
        if isinstance(h_id, dict):
            h_id = h_id.get("hospital_id") or h_id.get("id")

        if not isinstance(p_id, str) or p_id not in patients:
            violations.append(f"Hallucination: Unknown patient ID '{p_id}'.")
            continue
        if not isinstance(a_id, str) or a_id not in ambulances:
            violations.append(f"Hallucination: Unknown ambulance ID '{a_id}'.")
            continue
        if not isinstance(h_id, str) or h_id not in hospitals:
            violations.append(f"Hallucination: Unknown hospital ID '{h_id}'.")
            continue

        # Invariant 2: No duplicate assignments
        if p_id in assigned_patients:
            violations.append(f"DoubleAssignment: Patient '{p_id}' assigned multiple times.")
        assigned_patients.add(p_id)

        # Invariant 3: No double-booked ambulances
        if a_id in used_ambulances:
            violations.append(f"DoubleBooking: Ambulance '{a_id}' assigned to multiple patients.")
        used_ambulances.add(a_id)

        # Invariant 4 & 5: Hospital capacity limits
        pat = patients[p_id]
        hosp = hospitals[h_id]

        hosp["icu_beds"] -= 1
        if hosp["icu_beds"] < 0:
            icu_overflow_count += 1
            violations.append(f"ICUOverflow: Hospital '{h_id}' exceeded ICU bed capacity.")

        if pat.get("requires_ventilator", False):
            hosp["ventilators"] -= 1
            if hosp["ventilators"] < 0:
                vent_overflow_count += 1
                violations.append(
                    f"VentilatorOverflow: Hospital '{h_id}' exceeded ventilator capacity for critical patient '{p_id}'."
                )

        # Invariant 6: Golden-Hour SLA constraint
        eta_table = ambulances[a_id].get(p_id, {})
        eta = eta_table.get(h_id)
        if eta is None:
            violations.append(f"RouteUnavailable: No transit route between ambulance '{a_id}', patient '{p_id}', and hospital '{h_id}'.")
            continue

        window = pat["survival_window_mins"]
        if eta > window:
            breached_count += 1
            violations.append(
                f"SLABreach: Golden-Hour deadline exceeded for '{p_id}' (ETA {eta}m > Survival Window {window}m)."
            )
        else:
            sla_margins.append((window - eta) / float(window))

    # Invariant 2b: All patients must be dispatched
    missing_patients = set(patients.keys()) - assigned_patients
    for m_id in missing_patients:
        violations.append(f"UnassignedPatient: Critical patient '{m_id}' was left unassigned.")

    is_valid = len(violations) == 0
    total_count = len(patients)
    avg_margin = (sum(sla_margins) / total_count) if (is_valid and total_count > 0) else 0.0
    survived_count = max(0, len(sla_margins) - icu_overflow_count - vent_overflow_count)

    return {
        "valid": is_valid,
        "violations": violations,
        "sla_score": round(avg_margin, 4),
        "stats": {
            "assigned_count": len(assigned_patients),
            "unassigned_count": len(missing_patients),
            "total_patients": total_count,
            "breached_count": breached_count,
            "icu_overflow_count": icu_overflow_count,
            "vent_overflow_count": vent_overflow_count,
            "survived_count": survived_count,
        }
    }

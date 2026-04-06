from __future__ import annotations

ROOT_CAUSE_MAP = {
    "Freight": "Higher freight costs or shipment route changes",
    "Packaging": "Packaging material price inflation",
    "FX": "Foreign exchange movement",
    "Wages & Salaries": "Higher labor or overtime cost",
    "VLC": "Production inefficiency or increased wastage",
    "Supplier Change": "Supplier renegotiation or supplier switch",
}


def map_driver_to_root_cause(driver_name: str) -> str:
    if not driver_name:
        return "Mixed operational and commercial drivers"

    lower = driver_name.lower()
    for key, val in ROOT_CAUSE_MAP.items():
        if key.lower() in lower:
            return val

    return "Mixed operational and commercial drivers"


def enrich_root_cause(drivers: list[str]) -> list[str]:
    seen = []
    for d in drivers:
        cause = map_driver_to_root_cause(str(d))
        if cause not in seen:
            seen.append(cause)
    return seen

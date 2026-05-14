from __future__ import annotations

import argparse
import sys
from pathlib import Path



def _add_workspace_venv_site_packages() -> None:
    site_packages = Path(__file__).resolve().parent / ".venv_uv" / "Lib" / "site-packages"
    if not site_packages.exists():
        return

    site_packages_str = str(site_packages)
    if site_packages_str not in sys.path:
        sys.path.insert(0, site_packages_str)

_add_workspace_venv_site_packages()

from reports.generate_report import generate_monthly_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the monthly VILC report.")
    parser.add_argument("month", help="Month name or number, for example March or 3")
    parser.add_argument("year", help="Year, for example 2026")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = generate_monthly_report(month=args.month, year=args.year)
    print(f"HTML: {Path(result['html']).resolve()}")
    print(f"PNG:  {Path(result['png']).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
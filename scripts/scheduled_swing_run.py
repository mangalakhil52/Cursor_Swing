#!/usr/bin/env python3
"""Weekday 16:00 runner: scan, report, and update the Excel journal."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_analyst import load_dotenv_if_present  # noqa: E402
from src.constants import IST  # noqa: E402
from src.data_fetcher import DataFetcher  # noqa: E402
from src.scanner import SwingScanner, format_report  # noqa: E402
from src.trade_tracker import TradeTracker  # noqa: E402
from src.universe import load_symbols  # noqa: E402


def _log(message: str) -> None:
    log_dir = PROJECT_ROOT / "reports" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line)
    with (log_dir / "scheduled.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Scheduled swing scan + Excel tracker")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run regardless of weekday/time and prior successful run",
    )
    args = parser.parse_args()

    now = datetime.now(IST)
    reports_dir = PROJECT_ROOT / "reports"
    workbook = reports_dir / "swing_performance.xlsx"
    tracker = TradeTracker(workbook, max_hold_sessions=10)

    if not args.force:
        if now.weekday() >= 5:
            _log("Skipped: weekend")
            return 0
        if now.time() < time(16, 0):
            _log("Skipped: before 16:00 IST")
            return 0

    if not args.force and tracker.already_ran_successfully(now.date()):
        _log("Skipped: today's successful scan is already recorded")
        return 0

    started = now
    report_path = reports_dir / f"{now:%Y-%m-%d}_swing_report.txt"

    try:
        load_dotenv_if_present(str(PROJECT_ROOT / ".env"))
        with (PROJECT_ROOT / "config.yaml").open(encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        symbols = load_symbols(config, base_dir=PROJECT_ROOT)
        fetcher = DataFetcher(config["market"]["exchange_suffix"])

        _log("Updating outcomes for existing tracked trades")
        updated = tracker.update_active_trades(fetcher)
        _log(f"Updated {updated} active trade row(s)")

        scanner = SwingScanner(config, symbols=symbols)
        result = scanner.run(top_n=2, progress_cb=_log)
        report = format_report(result)
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")

        added = tracker.append_picks(result)
        tracker.log_run(
            run_date=now.date(),
            started_at=started,
            status="SUCCESS",
            picks=added,
            report_file=str(report_path),
            message=(
                f"Scan completed; {added} new pick(s); "
                f"{updated} existing row(s) refreshed"
            ),
        )
        _log(f"Success: {added} new pick(s); Excel: {workbook}")
        return 0
    except Exception as exc:  # noqa: BLE001
        tracker.log_run(
            run_date=now.date(),
            started_at=started,
            status="FAILED",
            picks=0,
            report_file=str(report_path),
            message=f"{type(exc).__name__}: {exc}",
        )
        _log(f"FAILED: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Indian Swing Trade Finder — two best trades of the week."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.ai_analyst import load_dotenv_if_present
from src.scanner import SwingScanner, format_report
from src.trade_tracker import TradeTracker
from src.universe import load_symbols


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find the two best NSE swing trades for the week (AI + rules).",
    )
    parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("-n", "--top", type=int, default=2, help="Max weekly picks (1 or 2)")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--no-ai", action="store_true", help="Quant shortlist only")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1

    load_dotenv_if_present(str(args.config.resolve().parent / ".env"))
    load_dotenv_if_present()

    config = load_config(args.config)
    if args.capital is not None:
        config["risk"]["capital"] = args.capital
    if args.no_ai:
        config.setdefault("ai", {})["enabled"] = False

    base_dir = args.config.resolve().parent
    try:
        symbols = load_symbols(config, base_dir=base_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    top_n = 1 if args.top <= 1 else 2
    ai_on = bool(config.get("ai", {}).get("enabled", True))
    print(f"Weekly swing scan — {len(symbols)} symbols | AI: {'ON' if ai_on else 'OFF'}")
    print(f"Goal: up to {top_n} best trade(s) of the week\n")

    def on_progress(msg: str) -> None:
        print(msg)

    scanner = SwingScanner(config, symbols=symbols)
    result = scanner.run(top_n=top_n, progress_cb=on_progress)
    report = format_report(result)
    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode("ascii", errors="replace").decode("ascii"))

    # Manual and scheduled runs share one journal. New symbols are appended
    # on every run; same-week duplicates remain a single tracked trade.
    try:
        tracker = TradeTracker(base_dir / "reports" / "swing_performance.xlsx")
        updated = tracker.update_active_trades(scanner.fetcher)
        added = tracker.append_picks(result)
        print(
            f"\nExcel journal updated: {added} new pick(s), "
            f"{updated} active trade(s) refreshed"
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"\nWarning: Excel journal could not be updated: {exc}",
            file=sys.stderr,
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"\nReport saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

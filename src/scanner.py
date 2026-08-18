"""Weekly swing scanner — finds up to two best trades of the week."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from tabulate import tabulate

from src.ai_analyst import AIAnalyst, AIVerdict, load_dotenv_if_present
from src.constants import CONVICTION_A, CONVICTION_B, IST, MODE_SWING
from src.data_fetcher import DataFetcher
from src.intelligence import build_market_context, swing_playbook
from src.risk import RiskManager, TradePlan
from src.scorer import AnalysisResult, TradeScorer, detect_nifty_bias
from src.universe import load_symbols


@dataclass
class TradeCandidate:
    analysis: AnalysisResult
    plan: TradePlan
    rank_score: float
    checklist: list[str] = field(default_factory=list)
    playbook: list[str] = field(default_factory=list)
    ai_selected: bool = False
    week_rank: int | None = None


class SwingScanner:
    """Alias kept for clarity; main entry uses this."""

    def __init__(self, config: dict, symbols: list[str] | None = None) -> None:
        self.config = config
        self.symbols = symbols or []
        self.fetcher = DataFetcher(config["market"]["exchange_suffix"])
        swing = config.get("swing", {})
        self.hold_horizon = str(swing.get("hold_horizon", "5-10 trading days"))
        self.risk = RiskManager(
            capital=config["risk"]["capital"],
            risk_per_trade_pct=config["risk"]["risk_per_trade_pct"],
            min_risk_reward=config["risk"]["min_risk_reward"],
            max_position_pct=config["risk"]["max_position_pct"],
            atr_stop_mult=float(swing.get("atr_stop_mult", 1.2)),
            atr_target_mult=float(swing.get("atr_target_mult", 2.2)),
            stop_buffer_atr=float(swing.get("stop_buffer_atr", 0.15)),
            min_stop_pct=float(swing.get("min_stop_pct", 1.2)),
            max_stop_pct=float(swing.get("max_stop_pct", 6.0)),
            max_target_pct=float(swing.get("max_target_pct", 12.0)),
        )

    def run(self, top_n: int = 2, progress_cb=None, force_mode: str | None = None) -> dict:
        # force_mode ignored — product is swing-only now
        _ = force_mode
        now = datetime.now(IST)
        symbols = self.symbols or load_symbols(self.config)
        top_n = min(max(top_n, 1), 2)  # hard-cap: two best of the week

        def progress(msg: str) -> None:
            if progress_cb:
                progress_cb(msg)

        progress("Mode: WEEKLY SWING — selecting up to 2 best trades of the week")
        progress(f"Loading daily data for {len(symbols)} NSE symbols...")

        benchmark = self.fetcher.fetch_benchmark(self.config["market"]["benchmark"])
        # Top up latest session when Yahoo daily lags
        try:
            nifty_intra = self.fetcher.fetch_intraday(self.config["market"]["benchmark"])
            benchmark, _ = self.fetcher.merge_latest_session(
                benchmark, nifty_intra if not nifty_intra.empty else None
            )
        except Exception:  # noqa: BLE001
            pass

        nifty_bias = detect_nifty_bias(benchmark)
        market = build_market_context(benchmark, nifty_bias)
        nifty_closes = benchmark["close"].astype(float) if not benchmark.empty else None

        daily_map = self.fetcher.fetch_daily_batch(symbols)
        progress(f"Daily ready for {len(daily_map)}/{len(symbols)}. Scoring swing setups...")

        # Yahoo's daily row often remains NaN for hours after NSE close.
        # Rebuild today's daily candle from 15-minute bars so the 16:00 run
        # analyzes the session that just closed, not yesterday's candle.
        progress("Refreshing the latest session from intraday bars...")
        latest_intraday = self.fetcher.fetch_intraday_batch(
            symbols,
            period="1d",
            chunk_size=80,
        )
        progress(
            f"Latest-session data ready for {len(latest_intraday)}/{len(symbols)}."
        )

        scorer = TradeScorer(self.config, market, nifty_closes=nifty_closes, mode=MODE_SWING)
        candidates: list[TradeCandidate] = []
        skipped: list[str] = []
        dead_count = 0
        ai_enabled = bool(self.config.get("ai", {}).get("enabled", True))
        min_score = float(self.config["scoring"]["min_score"])
        if ai_enabled:
            min_score = min(min_score, float(self.config.get("ai", {}).get("shortlist_min_score", 55)))
        allow_c = bool(self.config.get("intelligence", {}).get("allow_conviction_c", False)) or ai_enabled

        for symbol in symbols:
            daily = daily_map.get(symbol)
            if daily is None:
                skipped.append(f"{symbol}: no daily data")
                continue
            try:
                snapshot = self.fetcher.snapshot_from_daily(
                    symbol,
                    daily,
                    latest_intraday.get(symbol),
                )
                if snapshot is None:
                    skipped.append(f"{symbol}: insufficient history")
                    continue
                analysis = scorer.analyze(snapshot)
                if analysis is None:
                    dead_count += 1
                    skipped.append(f"{symbol}: no swing edge")
                    continue
                if analysis.score < min_score:
                    skipped.append(f"{symbol}: score {analysis.score}")
                    continue
                if analysis.conviction == "C" and not allow_c:
                    skipped.append(f"{symbol}: conviction C")
                    continue

                plan = self.risk.build_plan(
                    symbol=analysis.symbol,
                    direction=analysis.direction,
                    entry=analysis.entry,
                    trigger=analysis.trigger,
                    atr_value=analysis.atr_value,
                    support=analysis.support,
                    resistance=analysis.resistance,
                    hold_until=self.hold_horizon,
                )
                if plan is None:
                    skipped.append(f"{symbol}: sizing failed")
                    continue

                playbook = swing_playbook(
                    analysis.direction,
                    plan.entry,
                    plan.stop_loss,
                    plan.target_1,
                    plan.target,
                    plan.trigger,
                    analysis.intel,
                    self.hold_horizon,
                )
                checklist = [
                    f"Weekly swing | Hold {self.hold_horizon}",
                    f"Conviction {analysis.conviction} | {analysis.setup.replace('_', ' ')}",
                    f"{plan.direction} | R:R 1:{plan.risk_reward}",
                    f"RS 20d {analysis.relative_strength:+.1f}% vs Nifty",
                    "Manage on daily closes only — ignore intraday noise",
                ]
                rank_score = (
                    analysis.score
                    + {"A": 15, "B": 6, "C": 0}.get(analysis.conviction, 0)
                    + {"A+": 10, "A": 4, "B": 0}.get(
                        analysis.intel.setup_grade, 0
                    )
                    + analysis.confluence * 4
                    + abs(analysis.relative_strength)
                    + plan.risk_reward * 2
                )
                candidates.append(
                    TradeCandidate(
                        analysis=analysis,
                        plan=plan,
                        rank_score=rank_score,
                        checklist=checklist,
                        playbook=playbook,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                skipped.append(f"{symbol}: {exc}")

        candidates.sort(key=lambda c: c.rank_score, reverse=True)
        progress(f"Quant shortlist: {len(candidates)} swing candidates")

        ai_verdict: AIVerdict | None = None
        week_picks: list[TradeCandidate] = []

        if ai_enabled:
            load_dotenv_if_present()
            analyst = AIAnalyst(self.config)
            ok, detail = analyst.available()
            progress(f"AI swing analyst: {detail}")
            if ok and candidates:
                progress(f"Asking LLM for the 2 best trades of the week...")
                ai_verdict = analyst.analyze(
                    market=market,
                    timestamp=now.strftime("%Y-%m-%d %H:%M IST"),
                    candidates=candidates,
                    hold_horizon=self.hold_horizon,
                )
                week_picks = self._apply_ai(candidates, ai_verdict, top_n, analyst.require_approval)
                if ai_verdict.trade_approved:
                    syms = ", ".join(f"#{p.rank} {p.symbol}" for p in ai_verdict.picks)
                    progress(f"AI weekly picks: {syms}")
                else:
                    progress("AI: NO_TRADE this week — stay in cash")
            elif not candidates:
                progress("Empty shortlist — no swing trades")
            elif analyst.require_approval:
                progress(f"AI required but unavailable — staying flat ({detail})")
                week_picks = []
            else:
                week_picks = candidates[:top_n]
                for i, c in enumerate(week_picks, 1):
                    c.week_rank = i
        else:
            week_picks = [
                c for c in candidates if c.analysis.conviction in (CONVICTION_A, CONVICTION_B)
            ][:top_n] or candidates[:top_n]
            for i, c in enumerate(week_picks, 1):
                c.week_rank = i

        session_dates = [c.analysis.session_date for c in candidates if c.analysis.session_date]
        session_date = max(session_dates) if session_dates else None

        return {
            "timestamp": now.strftime("%Y-%m-%d %H:%M IST"),
            "mode": MODE_SWING,
            "hold_horizon": self.hold_horizon,
            "nifty_bias": market.bias,
            "market": market,
            "universe_size": len(symbols),
            "scanned": len(daily_map),
            "dead_filtered": dead_count,
            "session_date": session_date,
            "session_age_days": (now.date() - session_date).days if session_date else None,
            "best_trade": week_picks[0] if week_picks else None,
            "top_trades": week_picks,
            "all_candidates": candidates,
            "skipped": skipped,
            "ai_verdict": ai_verdict,
        }

    @staticmethod
    def _apply_ai(
        shortlist: list[TradeCandidate],
        verdict: AIVerdict,
        top_n: int,
        require_approval: bool,
    ) -> list[TradeCandidate]:
        by_symbol = {c.analysis.symbol: c for c in shortlist}
        if not verdict.trade_approved:
            return [] if require_approval else shortlist[:top_n]

        picks: list[TradeCandidate] = []
        for ai_pick in verdict.picks[:top_n]:
            chosen = by_symbol.get(ai_pick.symbol)
            if not chosen:
                continue
            chosen.ai_selected = True
            chosen.week_rank = ai_pick.rank
            if ai_pick.thesis:
                chosen.analysis.thesis = ai_pick.thesis
            if ai_pick.conviction:
                chosen.analysis.conviction = ai_pick.conviction
            if ai_pick.why_this:
                chosen.analysis.reasons = ai_pick.why_this + chosen.analysis.reasons
            if ai_pick.invalidation:
                chosen.analysis.risks = ai_pick.invalidation + chosen.analysis.risks
            if ai_pick.playbook:
                chosen.playbook = ai_pick.playbook
            chosen.rank_score = 200 - ai_pick.rank * 10 + ai_pick.ai_score
            picks.append(chosen)
        return picks


# Backwards-compatible name used by main.py
IntradayScanner = SwingScanner


def format_report(result: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("  INDIAN SWING TRADE FINDER — TWO BEST OF THE WEEK")
    lines.append("=" * 78)
    lines.append(f"  Scan time : {result['timestamp']}")
    lines.append(f"  Mode      : WEEKLY SWING (hold {result.get('hold_horizon', '5-10 days')})")
    lines.append(f"  Nifty bias: {result['nifty_bias']}")

    market = result.get("market")
    if market:
        lines.append(
            f"  Nifty     : day {market.day_change_pct:+.2f}% | week {market.week_change_pct:+.2f}% | "
            f"{'above' if market.above_ema else 'below'} 21 EMA"
        )

    ai_verdict = result.get("ai_verdict")
    if ai_verdict is not None:
        lines.append(f"  AI engine : {ai_verdict.provider}/{ai_verdict.model} | {ai_verdict.decision}")
        if ai_verdict.market_read:
            lines.append(f"  AI week   : {ai_verdict.market_read}")
        if ai_verdict.error:
            lines.append(f"  AI error  : {ai_verdict.error}")

    lines.append(
        f"  Universe  : {result.get('scanned', 0)}/{result.get('universe_size', 0)} scanned | "
        f"filtered {result.get('dead_filtered', 0)}"
    )
    if result.get("session_date"):
        lines.append(f"  Data as of: {result['session_date']}")
    lines.append("")

    picks = result.get("top_trades") or []
    if not picks:
        lines.append("  NO SWING TRADES THIS WEEK — stay in cash.")
        if ai_verdict is not None and ai_verdict.thesis:
            lines.append(f"  AI reason: {ai_verdict.thesis}")
        if ai_verdict is not None and ai_verdict.rejected:
            lines.append("")
            lines.append("  AI rejected:")
            for item in ai_verdict.rejected[:8]:
                lines.append(f"    - {item.get('symbol')}: {item.get('reason')}")
        lines.append("=" * 78)
        return "\n".join(lines)

    for c in picks:
        a, p = c.analysis, c.plan
        intel = a.intel
        rank = c.week_rank or (picks.index(c) + 1)
        tag = "AI-SELECTED" if c.ai_selected else "QUANT"
        lines.append("-" * 78)
        lines.append(f"  WEEKLY PICK #{rank}  |  Conviction {a.conviction}  |  {tag}")
        lines.append(f"  Symbol     : {a.symbol}")
        lines.append(f"  Setup      : {a.setup.replace('_', ' ').title()}")
        lines.append(f"  Video grade: {intel.setup_grade} ({'structure break' if intel.breaks_structure else 'displacement'})")
        lines.append(f"  Direction  : {p.direction}")
        lines.append(f"  Hold       : {p.hold_until}")
        lines.append(f"  Score      : {a.score}/100")
        lines.append("")
        lines.append("  THESIS")
        lines.append(f"  {a.thesis}")
        lines.append("")
        lines.append("  LEVELS (swing — daily timeframe)")
        lines.append(f"  Entry/Trigger : INR {p.entry:,.2f}")
        lines.append(f"  Stop loss     : INR {p.stop_loss:,.2f} ({p.stop_distance_pct:.2f}%)")
        lines.append(f"  Target 1      : INR {p.target_1:,.2f} (book 40-50%)")
        lines.append(f"  Target 2      : INR {p.target:,.2f} (trail balance)")
        lines.append(f"  R:R           : 1:{p.risk_reward}  |  Expected move {p.expected_move_pct:.2f}%")
        lines.append(f"  Level logic   : {p.level_basis}")
        lines.append(f"  Quantity      : {p.quantity}  |  Risk {p.risk_pct_of_capital}% of capital")
        lines.append("")
        lines.append("  SWING SNAPSHOT")
        lines.append(
            f"  RS 5d {intel.rs_5d:+.1f}% | RS 20d {intel.rs_20d:+.1f}% | RSI {a.rsi} | Vol {a.volume_ratio:.2f}x"
        )
        lines.append(
            f"  Trend {intel.trend_quality:.0f} | Pullback {intel.pullback_quality:.0f} | "
            f"Breakout {intel.breakout_quality:.0f} | ATR {intel.atr_pct:.2f}%"
        )
        lines.append(
            f"  vs 21 EMA {intel.dist_from_ema_pct:+.1f}% | vs 20d high {intel.dist_from_high_20d_pct:+.1f}%"
        )
        lines.append(
            f"  Fair value INR {intel.fair_value:,.2f} | distance "
            f"{intel.fair_value_distance_atr:+.2f} ATR "
            f"({intel.fair_value_distance_pct:+.2f}%)"
        )
        lines.append(
            f"  Displacement {intel.displacement_ratio:.2f}x previous body | "
            f"body/range {intel.body_to_range:.0%} | "
            f"{'BOS confirmed' if intel.breaks_structure else 'no BOS'}"
        )
        lines.append("")
        lines.append("  PLAYBOOK")
        for i, step in enumerate(c.playbook, 1):
            lines.append(f"  {i}. {step}")
        lines.append("")
        lines.append("  WHY")
        for reason in a.reasons[:6]:
            lines.append(f"    + {reason}")
        lines.append("  RISKS")
        for risk in a.risks[:5]:
            lines.append(f"    ! {risk}")
        lines.append("")

    if len(picks) == 1:
        lines.append("  (Only one high-quality swing found — do not force a second.)")
        lines.append("")

    # Compact summary table
    lines.append("-" * 78)
    lines.append("  WEEK AT A GLANCE")
    table = []
    for c in picks:
        table.append([
            f"#{c.week_rank or '-'}",
            c.analysis.symbol,
            c.analysis.conviction,
            c.plan.direction,
            c.analysis.setup.replace("_", " ")[:18],
            f"{c.analysis.relative_strength:+.1f}%",
            f"INR {c.plan.entry:,.0f}",
            f"INR {c.plan.target:,.0f}",
        ])
    lines.append(
        tabulate(
            table,
            headers=["#", "Symbol", "Conv", "Side", "Setup", "RS20d", "Entry", "Target"],
            tablefmt="simple",
        )
    )
    lines.append("")
    lines.append("-" * 78)
    lines.append("  Run once or twice a week (Sun night / Mon morning ideal).")
    lines.append("  Educational tool only. Not SEBI-registered advice.")
    lines.append("=" * 78)
    return "\n".join(lines)

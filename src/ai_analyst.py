"""LLM swing analyst — picks the two best trades of the week (or none)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import requests


SYSTEM_PROMPT = """You are a senior Indian equity SWING trader (NSE cash) with 15+ years of experience.
Horizon: 5-10 trading days. You do NOT scalp. You do NOT force trades.

You receive Nifty context + a quantitative SHORTLIST with swing metrics
(20d relative strength, EMA trend, pullback/breakout quality, ATR, volume,
rolling fair value, displacement quality, structure break, and levels).

Adapted methodology from JJ Simon's fair-pricing video:
- Fair price is adapted from a session open to 20-day volume-weighted fair value
  because this scanner trades daily/weekly NSE swings, not 90-minute Nasdaq futures.
- A+ setup = body-dominant displacement candle that breaks recent structure.
- A setup = valid displacement (body larger than previous body).
- B setup = do not trade.
- Mean reversion: displacement points back toward fair value after >=0.8 ATR deviation.
- Continuation: displacement moves away from fair value with the established trend.

Your job this week:
- Select up to TWO best swing trades for the coming week.
- Prefer quality over quantity. If only one is great, return one. If none, return NO_TRADE.
- Favor: RS leaders, trend pullbacks to 21 EMA, clean breakouts with volume, market alignment.
- Strongly prefer A+ over A; never select grade B.
- For fair-value reversion, verify there is enough distance to fair value for >=1.5R.
- Reject: extended chases, dead RS, tight ATR, counter-trend hope trades, event-lottery names.
- Think in DAILY closes, not intraday noise.

Return ONLY valid JSON (no markdown):
{
  "decision": "TRADE" | "NO_TRADE",
  "market_read": "1-2 sentences on Nifty for the week",
  "picks": [
    {
      "rank": 1,
      "symbol": "SYMBOL",
      "direction": "LONG" | "SHORT",
      "conviction": "A" | "B",
      "ai_score": 0-100,
      "thesis": "2-4 sentences",
      "why_this": ["...", "..."],
      "invalidation": ["...", "..."],
      "playbook": ["step1", "step2", "step3", "step4"]
    }
  ],
  "rejected": [{"symbol": "X", "reason": "..."}]
}

Rules:
- picks length must be 0, 1, or 2 only.
- decision=NO_TRADE when picks is empty.
- Only use symbols from the provided shortlist.
- Rank 1 = best trade of the week, Rank 2 = second best.
"""


@dataclass
class AIPick:
    rank: int
    symbol: str
    direction: str | None
    conviction: str | None
    ai_score: float
    thesis: str
    why_this: list[str] = field(default_factory=list)
    invalidation: list[str] = field(default_factory=list)
    playbook: list[str] = field(default_factory=list)


@dataclass
class AIVerdict:
    decision: str
    picks: list[AIPick] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)
    market_read: str = ""
    provider: str = ""
    model: str = ""
    raw_text: str = ""
    error: str | None = None

    # Back-compat helpers used by report
    @property
    def selected_symbol(self) -> str | None:
        return self.picks[0].symbol if self.picks else None

    @property
    def thesis(self) -> str:
        if not self.picks:
            return "No swing edge this week — stay in cash."
        return self.picks[0].thesis

    @property
    def ai_score(self) -> float:
        return self.picks[0].ai_score if self.picks else 0.0

    @property
    def conviction(self) -> str | None:
        return self.picks[0].conviction if self.picks else None

    @property
    def trade_approved(self) -> bool:
        return self.decision == "TRADE" and bool(self.picks) and self.error is None


class AIAnalyst:
    def __init__(self, config: dict) -> None:
        ai = config.get("ai", {})
        self.enabled = bool(ai.get("enabled", True))
        self.provider = str(ai.get("provider", "openai")).lower()
        self.model = str(ai.get("model", "")).strip() or self._default_model()
        self.base_url = ai.get("base_url")
        self.temperature = float(ai.get("temperature", 0.2))
        self.timeout = int(ai.get("timeout_sec", 90))
        self.shortlist_size = int(ai.get("shortlist_size", 15))
        self.require_approval = bool(ai.get("require_ai_approval", True))
        self.max_picks = int(ai.get("max_picks", 2))
        self._api_key = self._resolve_api_key()

    def available(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "AI disabled in config"
        if self.provider == "ollama":
            return True, f"ollama/{self.model}"
        if not self._api_key:
            return False, (
                f"Missing API key for '{self.provider}'. "
                "Copy .env.example to .env and set a key."
            )
        return True, f"{self.provider}/{self.model}"

    def _default_model(self) -> str:
        return {
            "openai": "gpt-4o-mini",
            "groq": "llama-3.3-70b-versatile",
            "openrouter": "openai/gpt-4o-mini",
            "ollama": "llama3.2",
            "gemini": "gemini-2.0-flash",
        }.get(self.provider, "gpt-4o-mini")

    def _resolve_api_key(self) -> str | None:
        generic = os.getenv("TRADE_AI_API_KEY", "").strip()
        if generic:
            return generic
        env_map = {
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "ollama": "OLLAMA_API_KEY",
        }
        return os.getenv(env_map.get(self.provider, "OPENAI_API_KEY"), "").strip() or None

    def analyze(
        self,
        *,
        market: Any,
        timestamp: str,
        candidates: list[Any],
        hold_horizon: str,
    ) -> AIVerdict:
        ok, detail = self.available()
        if not ok:
            return AIVerdict(decision="NO_TRADE", market_read=detail, provider=self.provider, model=self.model, error=detail)

        shortlist = candidates[: self.shortlist_size]
        user_prompt = json.dumps(
            {
                "scan_time": timestamp,
                "style": "WEEKLY SWING — pick up to 2 best trades of the week",
                "hold_horizon": hold_horizon,
                "nifty": {
                    "bias": getattr(market, "bias", "NEUTRAL"),
                    "day_change_pct": getattr(market, "day_change_pct", 0),
                    "week_change_pct": getattr(market, "week_change_pct", 0),
                    "above_ema": getattr(market, "above_ema", False),
                    "atr_pct": getattr(market, "atr_pct", 0),
                },
                "max_picks": self.max_picks,
                "candidates": [candidate_to_dict(c) for c in shortlist],
            },
            indent=2,
            default=str,
        )

        try:
            raw = self._chat(SYSTEM_PROMPT, "Analyze this swing shortlist and return JSON only.\n\n" + user_prompt)
            verdict = parse_ai_verdict(raw, max_picks=self.max_picks)
            verdict.provider = self.provider
            verdict.model = self.model
            verdict.raw_text = raw
            allowed = {c.analysis.symbol for c in shortlist}
            verdict.picks = [p for p in verdict.picks if p.symbol in allowed][: self.max_picks]
            if not verdict.picks:
                verdict.decision = "NO_TRADE"
            else:
                verdict.decision = "TRADE"
                for i, p in enumerate(verdict.picks, 1):
                    p.rank = i
            return verdict
        except Exception as exc:  # noqa: BLE001
            return AIVerdict(
                decision="NO_TRADE",
                market_read=str(exc),
                provider=self.provider,
                model=self.model,
                error=str(exc),
            )

    def _chat(self, system: str, user: str) -> str:
        if self.provider == "gemini":
            return self._chat_gemini(system, user)
        return self._chat_openai_compatible(system, user)

    def _chat_openai_compatible(self, system: str, user: str) -> str:
        base = self.base_url or {
            "openai": "https://api.openai.com/v1",
            "groq": "https://api.groq.com/openai/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "ollama": "http://localhost:11434/v1",
        }.get(self.provider, "https://api.openai.com/v1")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://localhost/swing-trade-finder"
            headers["X-Title"] = "Indian Swing Trade Finder"

        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        if self.provider == "ollama":
            body.pop("response_format", None)

        url = f"{base.rstrip('/')}/chat/completions"
        resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
        if resp.status_code >= 400:
            body.pop("response_format", None)
            resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _chat_gemini(self, system: str, user: str) -> str:
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": self.temperature, "responseMimeType": "application/json"},
        }
        resp = requests.post(url, params={"key": self._api_key}, json=body, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def candidate_to_dict(candidate: Any) -> dict[str, Any]:
    a = candidate.analysis
    p = candidate.plan
    i = a.intel
    return {
        "symbol": a.symbol,
        "setup": a.setup,
        "direction": a.direction,
        "quant_score": a.score,
        "conviction_quant": a.conviction,
        "confluence": a.confluence,
        "rs_5d": i.rs_5d,
        "rs_20d": i.rs_20d,
        "trend_quality": i.trend_quality,
        "pullback_quality": i.pullback_quality,
        "breakout_quality": i.breakout_quality,
        "video_setup_grade": i.setup_grade,
        "fair_value": i.fair_value,
        "fair_value_distance_pct": i.fair_value_distance_pct,
        "fair_value_distance_atr": i.fair_value_distance_atr,
        "displacement_ratio": i.displacement_ratio,
        "body_to_range": i.body_to_range,
        "displacement_direction": i.displacement_direction,
        "breaks_5d_structure": i.breaks_structure,
        "atr_pct": i.atr_pct,
        "dist_from_ema_pct": i.dist_from_ema_pct,
        "dist_from_high_20d_pct": i.dist_from_high_20d_pct,
        "volume_ratio": a.volume_ratio,
        "rsi": a.rsi,
        "entry": p.entry,
        "stop_loss": p.stop_loss,
        "stop_distance_pct": p.stop_distance_pct,
        "target_1": p.target_1,
        "target_2": p.target,
        "risk_reward": p.risk_reward,
        "expected_move_pct": p.expected_move_pct,
        "level_basis": p.level_basis,
        "hold_until": p.hold_until,
        "thesis_quant": a.thesis,
        "reasons": a.reasons[:6],
        "risks": a.risks[:4],
    }


def parse_ai_verdict(raw: str, max_picks: int = 2) -> AIVerdict:
    data = _extract_json(raw)
    decision = str(data.get("decision", "NO_TRADE")).upper().strip()
    picks_raw = data.get("picks") or []
    if not isinstance(picks_raw, list):
        picks_raw = []

    # Legacy single-symbol schema fallback
    if not picks_raw and data.get("selected_symbol"):
        picks_raw = [{
            "rank": 1,
            "symbol": data.get("selected_symbol"),
            "direction": data.get("direction"),
            "conviction": data.get("conviction"),
            "ai_score": data.get("ai_score", 0),
            "thesis": data.get("thesis", ""),
            "why_this": data.get("why_this_over_others") or data.get("why_this") or [],
            "invalidation": data.get("invalidation") or [],
            "playbook": data.get("playbook") or [],
        }]

    picks: list[AIPick] = []
    for idx, item in enumerate(picks_raw[:max_picks], 1):
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        direction = str(item.get("direction") or "").upper().strip() or None
        if direction not in ("LONG", "SHORT"):
            direction = None
        conviction = str(item.get("conviction") or "").upper().strip() or None
        if conviction not in ("A", "B", "C"):
            conviction = None
        picks.append(
            AIPick(
                rank=int(item.get("rank") or idx),
                symbol=symbol,
                direction=direction,
                conviction=conviction,
                ai_score=float(item.get("ai_score") or 0),
                thesis=str(item.get("thesis") or "").strip(),
                why_this=[str(x) for x in (item.get("why_this") or [])][:6],
                invalidation=[str(x) for x in (item.get("invalidation") or [])][:6],
                playbook=[str(x) for x in (item.get("playbook") or [])][:8],
            )
        )

    if decision not in ("TRADE", "NO_TRADE"):
        decision = "TRADE" if picks else "NO_TRADE"
    if not picks:
        decision = "NO_TRADE"

    rejected = data.get("rejected") or []
    if not isinstance(rejected, list):
        rejected = []

    return AIVerdict(
        decision=decision,
        picks=picks,
        rejected=[
            {"symbol": str(r.get("symbol", "")), "reason": str(r.get("reason", ""))}
            for r in rejected if isinstance(r, dict)
        ][:10],
        market_read=str(data.get("market_read") or "").strip(),
    )


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("AI response did not contain JSON")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("AI JSON root must be an object")
    return data


def load_dotenv_if_present(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

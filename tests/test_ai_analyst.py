"""Tests for LLM swing verdict parsing (no live API)."""

from src.ai_analyst import parse_ai_verdict


def test_parse_two_weekly_picks():
    raw = """
    {
      "decision": "TRADE",
      "market_read": "Nifty constructive for swing longs",
      "picks": [
        {
          "rank": 1,
          "symbol": "reliance",
          "direction": "LONG",
          "conviction": "A",
          "ai_score": 88,
          "thesis": "RS leader pullback to 21 EMA.",
          "why_this": ["Best RS", "Clean structure"],
          "invalidation": ["Close below 21 EMA"],
          "playbook": ["Buy pullback", "Stop under swing low", "Hold 1-2 weeks"]
        },
        {
          "rank": 2,
          "symbol": "TCS",
          "direction": "LONG",
          "conviction": "B",
          "ai_score": 74,
          "thesis": "Base break with volume.",
          "why_this": ["Breakout"],
          "invalidation": ["Failed breakout"],
          "playbook": ["Buy hold above breakout"]
        }
      ],
      "rejected": [{"symbol": "INFY", "reason": "Weak RS"}]
    }
    """
    v = parse_ai_verdict(raw, max_picks=2)
    assert v.decision == "TRADE"
    assert v.trade_approved is True
    assert len(v.picks) == 2
    assert v.picks[0].symbol == "RELIANCE"
    assert v.picks[1].symbol == "TCS"
    assert v.selected_symbol == "RELIANCE"


def test_parse_no_trade():
    raw = '{"decision":"NO_TRADE","picks":[],"market_read":"Chop","rejected":[]}'
    v = parse_ai_verdict(raw)
    assert v.decision == "NO_TRADE"
    assert v.trade_approved is False
    assert v.picks == []


def test_parse_legacy_single_symbol():
    raw = '{"decision":"TRADE","selected_symbol":"SBIN","direction":"LONG","conviction":"B","ai_score":70,"thesis":"OK"}'
    v = parse_ai_verdict(raw)
    assert v.trade_approved is True
    assert v.picks[0].symbol == "SBIN"

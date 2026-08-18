"""Tests for realistic, structure-first swing levels."""

from src.constants import DIRECTION_LONG, DIRECTION_SHORT
from src.risk import RiskManager


def _manager() -> RiskManager:
    return RiskManager(
        capital=100_000,
        risk_per_trade_pct=1.0,
        min_risk_reward=1.5,
        max_position_pct=25,
        atr_stop_mult=1.2,
        atr_target_mult=2.2,
        stop_buffer_atr=0.15,
        min_stop_pct=1.2,
        max_stop_pct=6.0,
        max_target_pct=12.0,
    )


def test_long_uses_nearby_structure_and_two_targets():
    plan = _manager().build_plan(
        symbol="TEST",
        direction=DIRECTION_LONG,
        entry=100,
        trigger=100,
        atr_value=2,
        support=98.5,
        resistance=108,
    )
    assert plan is not None
    assert plan.stop_loss == 97.6
    assert plan.target_1 == 102.4
    assert plan.target == 104.4
    assert plan.risk_reward >= 1.5
    assert plan.stop_distance_pct == 2.4


def test_rejects_trade_when_resistance_is_too_close():
    plan = _manager().build_plan(
        symbol="TEST",
        direction=DIRECTION_LONG,
        entry=100,
        trigger=100,
        atr_value=2,
        support=98.5,
        resistance=102,
    )
    assert plan is None


def test_rejects_trade_when_real_stop_is_too_far():
    plan = _manager().build_plan(
        symbol="TEST",
        direction=DIRECTION_LONG,
        entry=100,
        trigger=100,
        atr_value=2,
        support=90,
        resistance=110,
    )
    assert plan is None


def test_short_plan_is_symmetric():
    plan = _manager().build_plan(
        symbol="TEST",
        direction=DIRECTION_SHORT,
        entry=100,
        trigger=100,
        atr_value=2,
        support=92,
        resistance=101.5,
    )
    assert plan is not None
    assert plan.stop_loss == 102.4
    assert plan.target_1 == 97.6
    assert plan.target == 95.6
    assert plan.risk_reward >= 1.5

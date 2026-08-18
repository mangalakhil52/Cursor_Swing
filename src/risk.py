"""Structure-first swing levels and position sizing."""

from __future__ import annotations

from dataclasses import dataclass

from src.constants import DIRECTION_LONG, DIRECTION_SHORT, MODE_SWING


@dataclass
class TradePlan:
    symbol: str
    direction: str
    entry: float
    trigger: float
    stop_loss: float
    target_1: float
    target: float
    risk_reward: float
    quantity: int
    risk_amount: float
    position_value: float
    risk_pct_of_capital: float
    expected_move_pct: float
    stop_distance_pct: float
    level_basis: str
    mode: str = MODE_SWING
    hold_until: str = "5-10 trading days"


class RiskManager:
    """Build realistic levels; reject trades that need invented targets."""

    def __init__(
        self,
        capital: float,
        risk_per_trade_pct: float,
        min_risk_reward: float,
        max_position_pct: float,
        atr_stop_mult: float = 1.2,
        atr_target_mult: float = 2.2,
        stop_buffer_atr: float = 0.15,
        min_stop_pct: float = 1.2,
        max_stop_pct: float = 6.0,
        max_target_pct: float = 12.0,
    ) -> None:
        self.capital = capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.min_risk_reward = min_risk_reward
        self.max_position_pct = max_position_pct
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.stop_buffer_atr = stop_buffer_atr
        self.min_stop_pct = min_stop_pct
        self.max_stop_pct = max_stop_pct
        self.max_target_pct = max_target_pct

    def build_plan(
        self,
        symbol: str,
        direction: str,
        entry: float,
        trigger: float,
        atr_value: float,
        support: float,
        resistance: float,
        hold_until: str = "5-10 trading days",
        **_ignored,
    ) -> TradePlan | None:
        if atr_value <= 0 or entry <= 0:
            return None

        planned = trigger if trigger and trigger > 0 else entry
        min_stop_distance = max(
            atr_value * self.atr_stop_mult,
            planned * (self.min_stop_pct / 100),
        )
        max_stop_distance = planned * (self.max_stop_pct / 100)
        target_cap_distance = planned * (self.max_target_pct / 100)
        buffer = atr_value * self.stop_buffer_atr

        if direction == DIRECTION_LONG:
            if support and support < planned:
                structural_stop = support - buffer
                structural_distance = planned - structural_stop
                if structural_distance > max_stop_distance:
                    return None
                stop_loss = min(structural_stop, planned - min_stop_distance)
                level_basis = "nearest support + ATR buffer"
            else:
                stop_loss = planned - min_stop_distance
                level_basis = "ATR fallback (no valid nearby support)"
            risk = planned - stop_loss
            projected_target = planned + min(
                atr_value * self.atr_target_mult,
                target_cap_distance,
            )
            if resistance and resistance > planned:
                target = min(projected_target, resistance)
                level_basis += "; target at nearest resistance"
            else:
                target = projected_target
                level_basis += "; target at ATR projection"
            target_1 = min(planned + risk, target)
        else:
            if resistance and resistance > planned:
                structural_stop = resistance + buffer
                structural_distance = structural_stop - planned
                if structural_distance > max_stop_distance:
                    return None
                stop_loss = max(structural_stop, planned + min_stop_distance)
                level_basis = "nearest resistance + ATR buffer"
            else:
                stop_loss = planned + min_stop_distance
                level_basis = "ATR fallback (no valid nearby resistance)"
            risk = stop_loss - planned
            projected_target = planned - min(
                atr_value * self.atr_target_mult,
                target_cap_distance,
            )
            if support and support < planned:
                target = max(projected_target, support)
                level_basis += "; target at nearest support"
            else:
                target = projected_target
                level_basis += "; target at ATR projection"
            target_1 = max(planned - risk, target)

        if risk <= 0:
            return None

        rr = abs(target - planned) / risk
        if rr < self.min_risk_reward:
            # Do not stretch the target just to make the trade qualify.
            return None

        max_risk_amt = self.capital * (self.risk_per_trade_pct / 100)
        qty = int(max_risk_amt // risk)
        if qty < 1:
            return None

        pos = qty * planned
        max_pos = self.capital * (self.max_position_pct / 100)
        if pos > max_pos:
            qty = int(max_pos // planned)
            if qty < 1:
                return None
            pos = qty * planned

        risk_amt = qty * risk
        return TradePlan(
            symbol=symbol,
            direction=direction,
            entry=round(planned, 2),
            trigger=round(planned, 2),
            stop_loss=round(stop_loss, 2),
            target_1=round(target_1, 2),
            target=round(target, 2),
            risk_reward=round(rr, 2),
            quantity=qty,
            risk_amount=round(risk_amt, 2),
            position_value=round(pos, 2),
            risk_pct_of_capital=round((risk_amt / self.capital) * 100, 2),
            expected_move_pct=round(abs(target - planned) / planned * 100, 2),
            stop_distance_pct=round(risk / planned * 100, 2),
            level_basis=level_basis,
            mode=MODE_SWING,
            hold_until=hold_until,
        )

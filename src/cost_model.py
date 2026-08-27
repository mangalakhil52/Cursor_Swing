"""Conservative Indian cash/F&O swing transaction-cost model."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class CostConfig:
    brokerage_bps:float=2.0
    slippage_bps:float=10.0
    stamp_duty_bps:float=1.5
    exchange_txn_bps:float=.5
    gst_bps:float=3.5
    sebi_bps:float=.01
    stt_bps:float=10.0

def round_trip_cost_bps(cfg:CostConfig=CostConfig(), *, short_fno:bool=False)->float:
    """Conservative round-trip basis-point estimate; STT differs by direction/instrument."""
    stt = cfg.stt_bps if not short_fno else cfg.stt_bps*.25
    return 2*(cfg.brokerage_bps+cfg.slippage_bps+cfg.exchange_txn_bps+cfg.gst_bps+cfg.sebi_bps)+cfg.stamp_duty_bps+stt

def cost_return(notional:float,cfg:CostConfig=CostConfig(),*,short_fno:bool=False)->float:
    return abs(notional)*round_trip_cost_bps(cfg,short_fno=short_fno)/10000.0

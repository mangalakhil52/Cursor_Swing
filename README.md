# Indian Swing Trade Finder

Finds the **two best NSE swing trades of the week** (5–10 trading day hold).

Flow:
1. Quantitative scan of **full NSE equity** (~2,080 EQ stocks; liquidity-filtered)
2. Real **LLM trader** picks up to **2** weekly ideas — or says **NO_TRADE**

> Educational tool only. Not SEBI-registered advice.

## Setup

```bash
cd intraday-trade-finder
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python scripts/refresh_universe.py --source nse_equity
```

Add one API key in `.env` (`OPENAI_API_KEY` / `GROQ_API_KEY` / `GEMINI_API_KEY`).

In `config.yaml` set matching provider, e.g. Groq:

```yaml
ai:
  provider: groq
  model: llama-3.3-70b-versatile
```

## Run (once or twice a week)

Best: **Sunday night / Monday morning**.

```bash
python main.py
python main.py --capital 200000
python main.py -o reports/week.txt
```

Output: **Weekly Pick #1** and **#2** with entry, structure-based stop,
partial target (T1), final target (T2), thesis, and playbook.

## Automatic weekday run and Excel tracking

Install the Windows task once:

```powershell
.\scripts\install_schedule.ps1
```

Task behavior:

- Runs Monday–Friday at 16:00 IST
- Also runs at login; the runner proceeds only when login is after 16:00
- Skips weekends, pre-16:00 logins, and duplicate successful runs
- Uses current-session 15-minute bars to rebuild Yahoo's delayed daily candle
- Writes daily reports under `reports/`
- Maintains `reports/swing_performance.xlsx`

Workbook sheets:

- `Picks`: entry, stop, T1/T2, status, exits, return, and R multiple
- `Runs`: every scheduled success/failure
- `Summary`: win rate, average return, average R, and cumulative results

Tracking is conservative: if a daily candle touches both stop and target and
the order is unknown, it counts the stop first.

## What changed vs intraday

| Old | New |
|-----|-----|
| Same-day / BTST | **Weekly swing** |
| One best trade / day | **Two best of the week** |
| ORB / VWAP / 15:15 exit | EMA pullback, RS leaders, breakouts |
| Tight intraday stops | Nearest structure + ATR buffer; unrealistic plans rejected |

## Level logic

- Stop: nearest valid support/resistance plus a small ATR buffer
- T1: 1R, where 40–50% can be booked
- T2: nearest real resistance/support, capped by a 2.2 ATR projection
- The scanner rejects a trade if the structural stop exceeds 6% or the genuine
  target cannot provide at least 1.5:1 R:R
- Targets are never stretched merely to make a setup qualify

## Linked-video methodology (adapted)

The linked video teaches a 90-minute Nasdaq prop-firm strategy. This project
uses its transferable ideas on a **daily NSE swing** timeframe:

- Session-open fair price → **20-day volume-weighted fair value**
- 5-minute displacement → **daily candle body larger than the previous body**
- A+ setup → displacement with a close through prior 5-day structure
- A setup → valid displacement without structure break
- B setup → rejected
- Mean reversion → daily displacement back toward fair value from at least
  0.8 ATR away
- Continuation → displacement away from fair value with the prevailing trend

The video's aggressive prop-account scaling and fixed Nasdaq point stops were
not copied because they are inappropriate for Indian cash-market swing trades.

## Universe options

| Source | Symbols | Command |
|--------|---------|---------|
| **NSE Equity EQ (default)** | ~2,080 | `python scripts/refresh_universe.py --source nse_equity` |
| Nifty Total Market | ~750 | `python scripts/refresh_universe.py --source nifty_total_market` |
| Nifty 500 | 500 | `python scripts/refresh_universe.py --source nifty500` |

Set `universe.list_file` in `config.yaml` to the matching CSV.
Liquidity filters still drop illiquid names before scoring.

## License

MIT — use at your own risk.

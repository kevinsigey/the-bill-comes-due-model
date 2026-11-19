# The Bill Comes Due — backward event study

Backward earnings event study behind the paper **The Bill Comes Due**
by Kevin Sigey:
https://kevinsigey.github.io/portfolio/the-bill-comes-due.html

167 earnings prints. The pattern: beats rose a mean of +1.17% and still
fell 47.1% of the time; misses fell a mean of -4.92% and fell 83.3% of
the time. Guidance and the forward file, not the headline beat or miss,
did the separating.

## Files
- `earnings_event_study.ipynb` / `1_earnings_event_study.ipynb` — the study notebooks
- `earnings_event_study.py` — script version
- `earnings_events.csv`, `earnings_events_capm.csv` — event data
- `per_ticker_summary.csv` — per-ticker results

## What is not here
The forward-file machinery that applies this pattern to new prints as
they report is a live trading edge and is withheld while it is being
graded. This repo is the backward study only: the historical evidence
the paper stands on.

# Contributing to Bitcoin Barometer

Thanks for your interest! Bug reports, ideas and pull requests are all welcome.

## Ground rules

1. **No look-ahead.** Any new feature must use only data that was available on the day it is used. Add a shift if a
   source publishes with a delay.
2. **Pre-register new signals.** Before testing an idea, open an issue with the hypothesis, the expected sign, the
   horizon and the data source. Results are added as a new dated batch in `confirm.py`; existing hypotheses are never edited.
3. **Same gate for everyone.** A new model enters the forecast only if it passes the same walk-forward test and
   2-year holdout exam as everything else.
4. **Free data only.** No API keys, no paid sources, so anyone can reproduce the results.
5. **Plain English.** Anything shown on the dashboard should be understandable without a finance degree.

## Development

```bash
pip install -r requirements.txt
python run.py          # updates data and rebuilds docs/index.html
```

Please keep pull requests focused, describe how you checked the change, and include before/after screenshots for
dashboard changes.

## Reporting bugs

Open an issue with what you expected, what happened, and (if relevant) the section of the dashboard and the date.

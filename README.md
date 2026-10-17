# A/B Testing for Conversion Optimization

A Python walkthrough of how to run and read an A/B test: simulate traffic, compute conversion rates with confidence intervals, test significance, and monitor results live without fooling yourself.

![Real-time monitoring](outputs/realtime_ab_test_monitoring.png)

## What it does
- Simulates 10,000 visitors per variant (A = 10%, B = 12% true conversion)
- 95% confidence intervals and a one-sided two-proportion z-test
- Live monitoring in batches of 100 visitors, with p-value and lift tracking
- **Peeking-problem demo**: an A/A test showing why checking p-values repeatedly breaks your results
- Streamlit app: plug in your own numbers and get a verdict

## Run it
```bash
pip install -r requirements.txt
python ab_test.py            # saves plots to outputs/
python ab_test.py --live     # adds the live-updating plot
streamlit run app.py         # calculator app
```

## What I learned
**Peeking inflates false positives.** In an A/A test (no real difference), testing once at the end gave a ~5% false positive rate. Checking after every batch pushed it to ~27%. So the monitoring section uses a Bonferroni-corrected threshold (α / number of looks). It's conservative, but valid.

Other takeaways:
- Decide sample size upfront, then test once.
- A significant p-value doesn't say how big the effect is. Read the CI too.
- Sequential methods (e.g. alpha spending, SPRT) are the next step for real-world monitoring.

## Stack
Python, NumPy, Pandas, Matplotlib, SciPy, Statsmodels, Streamlit

## Author
Rohit Giri · [Portfolio](https://rohiit.online) · [LinkedIn](https://linkedin.com/in/rohiitgoswami)

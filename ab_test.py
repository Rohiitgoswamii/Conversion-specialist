"""
A/B Testing for Conversion Optimization
Author: Rohit Giri

Simulates an A/B test, computes conversion rates with 95% CIs, runs a
two-proportion z-test, and monitors the test in "real time" with a
correction for repeated peeking.

Run:
    python ab_test.py          # saves plots to outputs/
    python ab_test.py --live   # also shows the live-updating plot
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

OUT_DIR = "outputs"
ALPHA = 0.05


def confidence_interval(conversions, total, confidence=0.95):
    """Normal-approximation CI for a proportion."""
    p = conversions / total
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * np.sqrt(p * (1 - p) / total)
    return p - margin, p + margin


def z_test(conv_a, n_a, conv_b, n_b, alternative="larger"):
    """Two-proportion z-test. Tests whether B's rate is larger than A's."""
    return proportions_ztest(
        np.array([conv_b, conv_a]), np.array([n_b, n_a]), alternative=alternative
    )


def simulate_fixed_test(n_a, n_b, cr_a, cr_b, rng):
    conv_a = rng.binomial(n_a, cr_a)
    conv_b = rng.binomial(n_b, cr_b)
    return conv_a, conv_b


def results_table(conv_a, n_a, conv_b, n_b):
    rows = []
    for name, c, n in [("A", conv_a, n_a), ("B", conv_b, n_b)]:
        lo, hi = confidence_interval(c, n)
        rows.append({
            "Variant": name, "Visitors": n, "Conversions": c,
            "Conversion Rate": f"{c / n * 100:.2f}%",
            "CI Lower": f"{lo * 100:.2f}%", "CI Upper": f"{hi * 100:.2f}%",
        })
    return pd.DataFrame(rows)


def plot_fixed_test(conv_a, n_a, conv_b, n_b, path):
    rates, errs = [], []
    for c, n in [(conv_a, n_a), (conv_b, n_b)]:
        lo, hi = confidence_interval(c, n)
        r = c / n
        rates.append(r * 100)
        errs.append([(r - lo) * 100, (hi - r) * 100])

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["A", "B"], rates, color=["#3498db", "#e74c3c"],
                  alpha=0.85, edgecolor="black")
    ax.errorbar(["A", "B"], rates, yerr=np.array(errs).T,
                fmt="none", color="black", capsize=10, capthick=2)
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.5, f"{r:.2f}%",
                ha="center", fontweight="bold")
    ax.set_ylabel("Conversion Rate (%)")
    ax.set_title("A/B Test: Conversion Rate with 95% CI")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_monitoring(cr_a, cr_b, rng, batch_size=100, num_batches=60, live=False):
    """
    Sequential monitoring: look at the data after every batch.

    Checking a p-value repeatedly inflates false positives (the "peeking
    problem"). To stay honest, each look uses a Bonferroni-adjusted
    threshold: alpha / num_batches. Conservative, but valid.
    """
    adj_alpha = ALPHA / num_batches
    tot = {"nA": 0, "nB": 0, "cA": 0, "cB": 0}
    history = {"batch": [], "p": [], "lift": []}

    if live:
        plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))

    for batch in range(1, num_batches + 1):
        tot["nA"] += batch_size
        tot["nB"] += batch_size
        tot["cA"] += rng.binomial(batch_size, cr_a)
        tot["cB"] += rng.binomial(batch_size, cr_b)

        rate_a = tot["cA"] / tot["nA"]
        rate_b = tot["cB"] / tot["nB"]
        lift = (rate_b - rate_a) / rate_a * 100 if rate_a > 0 else 0.0

        if tot["cA"] > 0 and tot["cB"] > 0:
            _, p = z_test(tot["cA"], tot["nA"], tot["cB"], tot["nB"])
        else:
            p = 1.0

        history["batch"].append(batch)
        history["p"].append(p)
        history["lift"].append(lift)

        if batch % 10 == 0:
            print(f"Batch {batch}/{num_batches} | A={rate_a*100:.2f}% "
                  f"B={rate_b*100:.2f}% | lift={lift:.2f}% | p={p:.4f}")

        if live:
            _draw_monitor(ax1, ax2, history, adj_alpha)
            plt.pause(0.05)

    _draw_monitor(ax1, ax2, history, adj_alpha)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/realtime_ab_test_monitoring.png", dpi=200)
    if live:
        plt.ioff()
        plt.show()
    plt.close(fig)

    return tot, history, adj_alpha


def _draw_monitor(ax1, ax2, h, adj_alpha):
    ax1.clear()
    ax1.plot(h["batch"], h["p"], "b-", linewidth=2, label="p-value")
    ax1.axhline(ALPHA, color="orange", linestyle="--", label=f"naive α = {ALPHA}")
    ax1.axhline(adj_alpha, color="red", linestyle="--",
                label=f"corrected α = {adj_alpha:.4f}")
    ax1.set_yscale("log")
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("p-value (log)")
    ax1.set_title("p-value over time")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    ax2.clear()
    ax2.plot(h["batch"], h["lift"], "g-", linewidth=2, label="Observed lift")
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xlabel("Batch")
    ax2.set_ylabel("Lift (%)")
    ax2.set_title("Conversion lift (B vs A)")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)


def peeking_demo(n_sims=2000, num_batches=60, batch_size=100, cr=0.10, seed=0):
    """
    A/A test (no real difference). How often does peeking at every batch
    'find' a significant result at alpha = 0.05? Should be 5% if you only
    look once; it's much higher if you peek.
    """
    rng = np.random.default_rng(seed)
    false_pos_peek = 0
    false_pos_once = 0
    for _ in range(n_sims):
        a = rng.binomial(batch_size, cr, size=num_batches).cumsum()
        b = rng.binomial(batch_size, cr, size=num_batches).cumsum()
        n = batch_size * np.arange(1, num_batches + 1)
        p_hat = (a + b) / (2 * n)
        se = np.sqrt(2 * p_hat * (1 - p_hat) / n)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(se > 0, (b / n - a / n) / se, 0)
        p = 1 - stats.norm.cdf(z)
        false_pos_peek += (p < ALPHA).any()
        false_pos_once += p[-1] < ALPHA
    return false_pos_once / n_sims, false_pos_peek / n_sims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="show live plot")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    # --- 1. Fixed-horizon test ---
    n_a = n_b = 10_000
    conv_a, conv_b = simulate_fixed_test(n_a, n_b, 0.10, 0.12, rng)
    print("=" * 60, "\nFIXED-SAMPLE TEST\n" + "=" * 60)
    print(results_table(conv_a, n_a, conv_b, n_b).to_string(index=False))

    plot_fixed_test(conv_a, n_a, conv_b, n_b, f"{OUT_DIR}/ab_test_visualization.png")

    z, p = z_test(conv_a, n_a, conv_b, n_b)
    print(f"\nz = {z:.4f}, p = {p:.6f}")
    print("Result:", "B is significantly better than A" if p < ALPHA
          else "No significant difference")

    # --- 2. Sequential monitoring with correction ---
    print("\n" + "=" * 60, "\nREAL-TIME MONITORING\n" + "=" * 60)
    tot, hist, adj_alpha = run_monitoring(0.10, 0.12, rng, live=args.live)
    final_p = hist["p"][-1]
    first_sig = next((b for b, pv in zip(hist["batch"], hist["p"])
                      if pv < adj_alpha), None)
    print(f"\nFinal p = {final_p:.6f} (corrected threshold {adj_alpha:.4f})")
    if first_sig:
        print(f"First crossed corrected threshold at batch {first_sig}")
    print("Conclusion:", "Implement B" if final_p < adj_alpha
          else "Keep testing or keep A")

    # --- 3. Why peeking is a problem ---
    print("\n" + "=" * 60, "\nPEEKING PROBLEM DEMO (A/A test)\n" + "=" * 60)
    once, peek = peeking_demo()
    print(f"False positive rate, one look at the end : {once*100:.1f}%")
    print(f"False positive rate, peeking every batch : {peek*100:.1f}%")


if __name__ == "__main__":
    main()

"""Streamlit A/B test calculator. Run: streamlit run app.py"""
import numpy as np
import streamlit as st
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

st.set_page_config(page_title="A/B Test Calculator", page_icon="📊")
st.title("📊 A/B Test Calculator")
st.caption("Enter visitors and conversions for each variant to check if B beats A.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Variant A")
    n_a = st.number_input("Visitors", min_value=1, value=10000, key="na")
    c_a = st.number_input("Conversions", min_value=0, value=1000, key="ca")
with col2:
    st.subheader("Variant B")
    n_b = st.number_input("Visitors", min_value=1, value=10000, key="nb")
    c_b = st.number_input("Conversions", min_value=0, value=1200, key="cb")

alpha = st.select_slider("Significance level (α)", [0.01, 0.05, 0.10], value=0.05)


def ci(c, n, conf=0.95):
    p = c / n
    m = stats.norm.ppf((1 + conf) / 2) * np.sqrt(p * (1 - p) / n)
    return max(0, p - m), min(1, p + m)


if st.button("Run test", type="primary"):
    if c_a > n_a or c_b > n_b:
        st.error("Conversions can't exceed visitors.")
    elif c_a == 0 and c_b == 0:
        st.error("Need at least one conversion.")
    else:
        rate_a, rate_b = c_a / n_a, c_b / n_b
        lo_a, hi_a = ci(c_a, n_a)
        lo_b, hi_b = ci(c_b, n_b)
        z, p = proportions_ztest([c_b, c_a], [n_b, n_a], alternative="larger")
        lift = (rate_b - rate_a) / rate_a * 100 if rate_a else float("nan")

        m1, m2, m3 = st.columns(3)
        m1.metric("A rate", f"{rate_a*100:.2f}%")
        m2.metric("B rate", f"{rate_b*100:.2f}%", f"{lift:.1f}% lift")
        m3.metric("p-value", f"{p:.4f}")

        st.write(f"**95% CI** → A: {lo_a*100:.2f}% – {hi_a*100:.2f}% | "
                 f"B: {lo_b*100:.2f}% – {hi_b*100:.2f}%")

        if p < alpha:
            st.success("B is significantly better than A.")
        else:
            st.warning("No significant difference detected.")
        st.caption("One-sided z-test (B > A). Decide your sample size upfront "
                   "and test once; don't check repeatedly.")

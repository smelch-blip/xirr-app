import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta

# ============================================================
# CONFIG
# ============================================================

REQUIRED_COLUMNS = ["Ticker", "Date", "Action", "Quantity", "Price", "Charges", "CMP"]
ACTION_ALLOWED = {"BUY", "SELL"}

st.set_page_config(page_title="Portfolio XIRR + Intrinsic + NIFTY", layout="wide")
st.title("📈 Stock-wise XIRR + Intrinsic Valuation + NIFTY Benchmark")

st.warning(
    "⚠️ Corporate actions (splits/bonus) and dividends are NOT auto-captured. "
    "If quantities are not adjusted, XIRR and intrinsic comparison may be skewed."
)

# ============================================================
# XIRR ENGINE
# ============================================================

def xnpv(rate, cashflows):
    t0 = min(dt for dt, _ in cashflows)
    return sum(amt / ((1 + rate) ** ((dt - t0).days / 365.25)) for dt, amt in cashflows)

def xirr(cashflows):
    if len(cashflows) < 2:
        return None
    amounts = [amt for _, amt in cashflows]
    if not (any(a < 0 for a in amounts) and any(a > 0 for a in amounts)):
        return None

    r = 0.15
    for _ in range(50):
        f = xnpv(r, cashflows)
        df = (xnpv(r + 1e-5, cashflows) - f) / 1e-5
        if abs(df) < 1e-10:
            break
        r -= f / df
        r = max(-0.99, min(r, 10))
    return r

# ============================================================
# NIFTY BENCHMARK
# ============================================================

def compute_nifty_xirr(portfolio_cf, valuation_date):
    start = min(d for d, _ in portfolio_cf)
    df = yf.download("^NSEI", start=start - timedelta(days=5),
                     end=valuation_date + timedelta(days=5), progress=False)
    if df.empty:
        return None

    prices = df["Close"]
    units = 0.0
    bench_cf = []

    for d, amt in portfolio_cf:
        if amt < 0:
            px = prices[prices.index <= pd.Timestamp(d)].iloc[-1]
            units += abs(amt) / px
            bench_cf.append((d, amt))

    final_px = prices[prices.index <= pd.Timestamp(valuation_date)].iloc[-1]
    bench_cf.append((valuation_date, units * final_px))
    return xirr(bench_cf)

# ============================================================
# INTRINSIC VALUATION (REDESIGNED)
# ============================================================

def classify_business(ticker):
    t = ticker.upper()
    if "BANK" in t:
        return "FINANCIAL"
    if any(x in t for x in ["IT", "TECH"]):
        return "ASSET_LIGHT"
    if any(x in t for x in ["STEEL", "METAL", "ENERGY", "POWER"]):
        return "CYCLICAL"
    return "GENERAL"

VALUATION_RULES = {
    "FINANCIAL": {"epv": 8, "justified": 14, "confidence": "High"},
    "ASSET_LIGHT": {"epv": 15, "justified": 30, "confidence": "High"},
    "CYCLICAL": {"epv": 6, "justified": 12, "confidence": "Low"},
    "GENERAL": {"epv": 10, "justified": 18, "confidence": "Medium"},
}

def intrinsic_range(cmp, business):
    if cmp <= 0:
        return None, None, None
    cfg = VALUATION_RULES[business]
    intr_min = cmp * cfg["epv"] / 15
    intr_max = cmp * cfg["justified"] / 15
    return round(intr_min, 2), round(intr_max, 2), cfg["confidence"]

def margin_of_safety(cmp, intr_min):
    if cmp <= 0 or intr_min <= 0:
        return None
    return round((intr_min - cmp) / intr_min * 100, 2)

# ============================================================
# HELPERS
# ============================================================

def parse_date(x):
    return pd.to_datetime(x, errors="coerce").date()

def clean_df(df):
    df = df.copy()[REQUIRED_COLUMNS]
    df["Ticker"] = df["Ticker"].str.upper().str.strip()
    df["Action"] = df["Action"].str.upper().str.strip()
    df["Date"] = df["Date"].apply(parse_date)
    for c in ["Quantity", "Price", "Charges", "CMP"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df[df["Action"].isin(ACTION_ALLOWED)]

# ============================================================
# UI
# ============================================================

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    raw = pd.read_csv(uploaded)
    if any(c not in raw.columns for c in REQUIRED_COLUMNS):
        st.error("CSV does not match required template.")
        st.stop()

    df = clean_df(raw)
    valuation_date = st.date_input("Valuation Date", value=date.today())

    cmp_df = df.groupby("Ticker")["CMP"].max().reset_index()
    cmp_edit = st.data_editor(cmp_df, hide_index=True)

    if st.button("🚀 Compute XIRR & Intrinsic"):
        cmp_map = dict(zip(cmp_edit["Ticker"], cmp_edit["CMP"]))
        rows = []
        portfolio_cf = []

        for tkr, g in df.groupby("Ticker"):
            qty = invested = realized = 0
            cashflows = []

            for _, r in g.iterrows():
                amt = -(r["Quantity"] * r["Price"] + r["Charges"]) if r["Action"] == "BUY" \
                      else (r["Quantity"] * r["Price"] - r["Charges"])
                qty += r["Quantity"] if r["Action"] == "BUY" else -r["Quantity"]
                invested += -amt if r["Action"] == "BUY" else 0
                realized += amt if r["Action"] == "SELL" else 0
                cashflows.append((r["Date"], amt))
                portfolio_cf.append((r["Date"], amt))

            cmp = cmp_map.get(tkr, 0)
            curr_val = qty * cmp
            if qty > 0:
                cashflows.append((valuation_date, curr_val))
                portfolio_cf.append((valuation_date, curr_val))

            r = xirr(cashflows)
            business = classify_business(tkr)
            intr_min, intr_max, conf = intrinsic_range(cmp, business) if qty > 0 else (None, None, None)
            mos = margin_of_safety(cmp, intr_min) if intr_min else None

            rows.append({
                "Ticker": tkr,
                "Holding Qty": qty,
                "CMP": cmp,
                "Current Value": curr_val,
                "Total Invested": invested,
                "Total Realized": realized,
                "XIRR %": None if r is None else round(r * 100, 2),
                "Intrinsic Range": f"{intr_min} – {intr_max}" if intr_min else "NA",
                "MoS %": mos,
                "Valuation Confidence": conf
            })

        stock_df = pd.DataFrame(rows)
        st.subheader("📌 Stock-wise XIRR + Intrinsic")
        st.dataframe(stock_df, use_container_width=True)

        px = xirr(portfolio_cf)
        nx = compute_nifty_xirr(portfolio_cf, valuation_date)

        st.subheader("📊 Portfolio vs NIFTY 50")
        if px and nx:
            st.metric("Portfolio XIRR", f"{px*100:.2f}%")
            st.metric("NIFTY XIRR", f"{nx*100:.2f}%")
            st.metric("Alpha", f"{(px-nx)*100:.2f}%")
        else:
            st.warning("Benchmark not computable today.")

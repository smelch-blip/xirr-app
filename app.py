import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date, timedelta

# ============================================================
# CONFIG
# ============================================================

REQUIRED_COLUMNS = ["Ticker", "Date", "Action", "Quantity", "Price", "Charges", "CMP"]
ACTION_ALLOWED = {"BUY", "SELL"}

st.set_page_config(page_title="Portfolio XIRR Analyzer", layout="wide")
st.title("📈 Stock-wise XIRR + Intrinsic Value + NIFTY Benchmark")

st.warning(
    "⚠️ Corporate actions (splits/bonus) and dividends are NOT auto-captured. "
    "If these apply and are not reflected in quantities/cashflows, XIRR may be skewed."
)

# ============================================================
# XIRR ENGINE
# ============================================================

def xnpv(rate, cashflows):
    if rate <= -0.999999:
        return np.inf
    t0 = min(dt for dt, _ in cashflows)
    total = 0.0
    for dt, amt in cashflows:
        years = (dt - t0).days / 365.25
        total += amt / ((1.0 + rate) ** years)
    return total

def xirr(cashflows):
    if len(cashflows) < 2:
        return None
    amounts = [amt for _, amt in cashflows]
    if not (any(a < 0 for a in amounts) and any(a > 0 for a in amounts)):
        return None

    r = 0.15
    for _ in range(50):
        f = xnpv(r, cashflows)
        dr = 1e-5
        f2 = xnpv(r + dr, cashflows)
        d = (f2 - f) / dr
        if abs(d) < 1e-12:
            break
        r_new = r - f / d
        if not np.isfinite(r_new):
            break
        r_new = max(-0.999, min(10.0, r_new))
        if abs(r_new - r) < 1e-8:
            return r_new
        r = r_new

    grid = np.concatenate([
        np.linspace(-0.9, 0.0, 30),
        np.linspace(0.0, 1.0, 50),
        np.linspace(1.0, 10.0, 50),
    ])
    prev_r = grid[0]
    prev_f = xnpv(prev_r, cashflows)
    for rr in grid[1:]:
        ff = xnpv(rr, cashflows)
        if np.isfinite(prev_f) and np.isfinite(ff) and (prev_f * ff < 0):
            a, b = prev_r, rr
            break
        prev_r, prev_f = rr, ff
    else:
        return None

    fa, fb = xnpv(a, cashflows), xnpv(b, cashflows)
    for _ in range(80):
        mid = (a + b) / 2
        fm = xnpv(mid, cashflows)
        if abs(fm) < 1e-8:
            return mid
        if fa * fm < 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return (a + b) / 2

# ============================================================
# BENCHMARK: NIFTY 50
# ============================================================

NIFTY_TICKER = "^NSEI"

def fetch_nifty_prices(start_date, end_date):
    df = yf.download(
        NIFTY_TICKER,
        start=start_date - timedelta(days=5),
        end=end_date + timedelta(days=5),
        progress=False
    )
    if df.empty:
        return None
    return df[["Close"]].dropna()

def nifty_price_on_or_before(price_df, target_date):
    eligible = price_df[price_df.index <= pd.Timestamp(target_date)]
    if eligible.empty:
        return None
    return float(eligible.iloc[-1]["Close"])

def compute_nifty_xirr(portfolio_cashflows, valuation_date):
    start_date = min(d for d, _ in portfolio_cashflows)
    prices = fetch_nifty_prices(start_date, valuation_date)
    if prices is None:
        return None

    units = 0.0
    cashflows = []

    for d, amt in portfolio_cashflows:
        if amt < 0:
            px = nifty_price_on_or_before(prices, d)
            if px is None:
                return None
            units += abs(amt) / px
            cashflows.append((d, amt))

    final_px = nifty_price_on_or_before(prices, valuation_date)
    if final_px is None:
        return None

    cashflows.append((valuation_date, units * final_px))
    return xirr(cashflows)

# ============================================================
# INTRINSIC VALUATION (STATIC)
# ============================================================

def classify_business(ticker):
    t = ticker.upper()
    if "BANK" in t:
        return "FINANCIAL"
    if any(x in t for x in ["IT", "TECH", "SOFT"]):
        return "ASSET_LIGHT"
    if any(x in t for x in ["STEEL", "METAL", "ENERGY", "POWER"]):
        return "CYCLICAL"
    return "GENERAL"

ASSUMPTIONS = {
    "FINANCIAL": (1.8, "Medium"),
    "ASSET_LIGHT": (25, "High"),
    "CYCLICAL": (12, "Low"),
    "GENERAL": (18, "Medium"),
}

def intrinsic_values(cmp, business):
    if cmp <= 0:
        return None, None, None, None
    mult, conf = ASSUMPTIONS[business]
    base = cmp * mult / 20
    low = base * 0.85
    high = base * 1.15
    return round(low,2), round(base,2), round(high,2), conf

def margin_of_safety(cmp, base):
    if cmp <= 0 or base <= 0:
        return None
    return round((base - cmp) / base * 100, 2)

# ============================================================
# HELPERS
# ============================================================

def parse_date(x):
    return pd.to_datetime(x, errors="coerce").date()

def clean_transactions(df):
    df = df.copy()[REQUIRED_COLUMNS]
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Action"] = df["Action"].astype(str).str.upper().str.strip()
    df["Date"] = df["Date"].apply(parse_date)
    for c in ["Quantity", "Price", "Charges", "CMP"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df = df[df["Action"].isin(ACTION_ALLOWED)]
    return df.sort_values(["Ticker", "Date"])

def build_cmp_table(df):
    return (
        df.groupby("Ticker")["CMP"]
        .max()
        .reset_index()
        .round(2)
    )

# ============================================================
# UI
# ============================================================

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    raw = pd.read_csv(uploaded)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    df = clean_transactions(raw)
    valuation_date = st.date_input("Valuation date", value=date.today())

    cmp_df = build_cmp_table(df)
    edited_cmp = st.data_editor(cmp_df, hide_index=True)

    if st.button("🚀 Compute XIRR"):
        cmp_map = dict(zip(edited_cmp["Ticker"], edited_cmp["CMP"]))
        stock_rows = []
        portfolio_cf = []

        for tkr, g in df.groupby("Ticker"):
            qty = invested = realized = 0.0
            cashflows = []

            for _, r in g.iterrows():
                amt = -(r["Quantity"] * r["Price"] + r["Charges"]) if r["Action"]=="BUY" else (r["Quantity"] * r["Price"] - r["Charges"])
                qty += r["Quantity"] if r["Action"]=="BUY" else -r["Quantity"]
                invested += -amt if r["Action"]=="BUY" else 0
                realized += amt if r["Action"]=="SELL" else 0
                cashflows.append((r["Date"], amt))
                portfolio_cf.append((r["Date"], amt))

            cmp_val = cmp_map.get(tkr, 0.0)
            current_val = qty * cmp_val
            if qty > 0:
                cashflows.append((valuation_date, current_val))
                portfolio_cf.append((valuation_date, current_val))

            r = xirr(cashflows)
            business = classify_business(tkr)
            il, ib, ih, conf = intrinsic_values(cmp_val, business) if qty > 0 else (None, None, None, None)
            mos = margin_of_safety(cmp_val, ib) if ib else None

            stock_rows.append({
                "Ticker": tkr,
                "Holding Qty": round(qty,2),
                "CMP": round(cmp_val,2),
                "Current Value": round(current_val,2),
                "Total Invested": round(invested,2),
                "Total Realized": round(realized,2),
                "XIRR %": None if r is None else round(r*100,2),
                "Intrinsic Low": il,
                "Intrinsic Base": ib,
                "Intrinsic High": ih,
                "MoS %": mos,
                "Valuation Confidence": conf
            })

        stock_df = pd.DataFrame(stock_rows)
        st.subheader("📌 Stock-wise XIRR + Intrinsic")
        st.dataframe(stock_df, use_container_width=True)

        portfolio_xirr = xirr(portfolio_cf)
        nifty_xirr = compute_nifty_xirr(portfolio_cf, valuation_date)

        st.subheader("📊 Portfolio vs NIFTY 50")
        if portfolio_xirr and nifty_xirr:
            st.metric("Portfolio XIRR", f"{portfolio_xirr*100:.2f}%")
            st.metric("NIFTY XIRR", f"{nifty_xirr*100:.2f}%")
            st.metric("Alpha", f"{(portfolio_xirr-nifty_xirr)*100:.2f}%")
        else:
            st.warning("Benchmark not computable")

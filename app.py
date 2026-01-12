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
st.title("📈 Stock-wise XIRR + Portfolio XIRR + NIFTY Benchmark")

st.warning(
    "⚠️ Corporate actions (splits/bonus) and dividends are NOT auto-captured in this version. "
    "If these apply to your holdings and are not reflected in quantities/cashflows, XIRR may be skewed."
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

    # Newton-Raphson
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

    # Bisection fallback
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
        progress=False,
        auto_adjust=False
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
    dates = [d for d, _ in portfolio_cashflows]
    start_date = min(dates)
    nifty_prices = fetch_nifty_prices(start_date, valuation_date)
    if nifty_prices is None:
        return None, None

    nifty_units = 0.0
    benchmark_cashflows = []

    for d, amt in portfolio_cashflows:
        if amt < 0:
            px = nifty_price_on_or_before(nifty_prices, d)
            if px is None:
                return None, None
            nifty_units += abs(amt) / px
            benchmark_cashflows.append((d, amt))

    final_px = nifty_price_on_or_before(nifty_prices, valuation_date)
    if final_px is None:
        return None, None

    benchmark_value = nifty_units * final_px
    benchmark_cashflows.append((valuation_date, benchmark_value))

    return xirr(benchmark_cashflows), benchmark_value

# ============================================================
# HELPERS
# ============================================================

def parse_date(x):
    return pd.to_datetime(x, errors="coerce").date()

def clean_transactions(df):
    df = df.copy()[REQUIRED_COLUMNS]
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Action"] = df["Action"].astype(str).str.strip().str.upper()
    df["Date"] = df["Date"].apply(parse_date)

    for col in ["Quantity", "Price", "Charges", "CMP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df[df["Action"].isin(ACTION_ALLOWED)]
    df = df[df["Date"].notna()]
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

def build_cmp_table(df):
    rows = []
    for tkr, g in df.groupby("Ticker"):
        nz = g["CMP"][g["CMP"] > 0]
        cmp_val = nz.iloc[-1] if not nz.empty else 0.0
        rows.append({"Ticker": tkr, "CMP": round(float(cmp_val), 2)})
    return pd.DataFrame(rows)

# ============================================================
# UI: TEMPLATE
# ============================================================

with st.expander("📄 Download upload template"):
    tmpl = pd.DataFrame([
        {"Ticker": "INFY", "Date": "2019-06-10", "Action": "BUY", "Quantity": 10, "Price": 700, "Charges": 0, "CMP": 0},
        {"Ticker": "INFY", "Date": "2021-05-20", "Action": "SELL", "Quantity": 3, "Price": 1200, "Charges": 0, "CMP": 0},
    ])
    st.dataframe(tmpl, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download CSV", tmpl.to_csv(index=False), "PortfolioImportTemplate.csv")

# ============================================================
# UI: UPLOAD
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
    edited_cmp = st.data_editor(
        cmp_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn(disabled=True),
            "CMP": st.column_config.NumberColumn(step=0.05)
        }
    )

    if st.button("🚀 Compute XIRR"):
        cmp_map = dict(zip(edited_cmp["Ticker"], edited_cmp["CMP"]))

        stock_rows = []
        portfolio_cashflows = []

        for tkr, g in df.groupby("Ticker"):
            qty = 0.0
            invested = 0.0
            realized = 0.0
            cashflows = []

            for _, r in g.iterrows():
                dt = r["Date"]
                q = r["Quantity"]
                p = r["Price"]
                ch = r["Charges"]

                if r["Action"] == "BUY":
                    amt = -(q * p + ch)
                    qty += q
                    invested += (q * p + ch)
                else:
                    amt = +(q * p - ch)
                    qty -= q
                    realized += (q * p - ch)

                cashflows.append((dt, amt))
                portfolio_cashflows.append((dt, amt))

            cmp_val = cmp_map.get(tkr, 0.0)
            current_value = qty * cmp_val
            if abs(current_value) > 0:
                cashflows.append((valuation_date, current_value))

            r = xirr(cashflows)

            stock_rows.append({
                "Ticker": tkr,
                "Holding Qty": round(qty, 2),
                "CMP": round(cmp_val, 2),
                "Current Value": round(current_value, 2),
                "Total Invested": round(invested, 2),
                "Total Realized": round(realized, 2),
                "XIRR %": None if r is None else round(r * 100, 2)
            })

        stock_df = pd.DataFrame(stock_rows)

        total_current_value = stock_df["Current Value"].sum()
        portfolio_cashflows.append((valuation_date, total_current_value))

        portfolio_xirr = xirr(portfolio_cashflows)
        nifty_xirr, _ = compute_nifty_xirr(portfolio_cashflows, valuation_date)

        st.subheader("📌 Stock-wise XIRR")
        st.dataframe(stock_df, use_container_width=True, hide_index=True)

        st.subheader("📊 Portfolio vs NIFTY 50")
        if portfolio_xirr is not None and nifty_xirr is not None:
            alpha = portfolio_xirr - nifty_xirr
            c1, c2, c3 = st.columns(3)
            c1.metric("Portfolio XIRR (%)", f"{portfolio_xirr*100:.2f}%")
            c2.metric("NIFTY XIRR (%)", f"{nifty_xirr*100:.2f}%")
            c3.metric("Alpha (%)", f"{alpha*100:.2f}%")
        else:
            st.warning("Benchmark XIRR could not be computed.")

        st.caption(
            "Benchmark uses NIFTY 50 price index (^NSEI). Dividends are excluded."
        )

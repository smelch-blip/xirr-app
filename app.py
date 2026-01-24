import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import yfinance as yf

# ============================================================
# CONFIG
# ============================================================
REQUIRED_COLUMNS = ["Ticker", "Date", "Action", "Quantity", "Price", "Charges", "CMP"]
ACTION_ALLOWED = {"BUY", "SELL"}

st.set_page_config(page_title="Portfolio XIRR + Intrinsic + NIFTY", layout="wide")
st.title("📈 Portfolio XIRR + Intrinsic Valuation + NIFTY 50 Benchmark")

st.warning(
    "⚠️ Corporate actions (splits/bonus/dividends) are NOT auto-captured. "
    "Ensure quantities and prices are adjusted manually."
)

# ============================================================
# HELPERS
# ============================================================
def parse_date(x):
    try:
        return pd.to_datetime(x).date()
    except:
        return None

def xnpv(rate, cashflows):
    t0 = min(d for d, _ in cashflows)
    return sum(a / ((1 + rate) ** ((d - t0).days / 365.25)) for d, a in cashflows)

def xirr(cashflows):
    if len(cashflows) < 2:
        return None
    if not (any(a < 0 for _, a in cashflows) and any(a > 0 for _, a in cashflows)):
        return None

    r = 0.15
    for _ in range(50):
        f = xnpv(r, cashflows)
        d = (xnpv(r + 1e-5, cashflows) - f) / 1e-5
        if abs(d) < 1e-10:
            break
        r -= f / d
        r = max(-0.99, min(r, 10))
    return r

# ============================================================
# INTRINSIC LOGIC (CLEAN & STABLE)
# ============================================================
def intrinsic_value(eps, bvps, roe, biz, band=12):
    """
    Returns: intrinsic_mid, intrinsic_low, intrinsic_high, confidence
    """
    if biz == "BANK":
        if bvps <= 0 or roe <= 0:
            return None, None, None, "Low"
        target_pb = min(2.0, max(0.8, roe / 0.13))
        mid = bvps * target_pb
        conf = "High"
    else:
        if eps <= 0:
            return None, None, None, "Low"
        multiple = {
            "ASSET_LIGHT": 22,
            "CYCLICAL": 10,
            "GENERAL": 15
        }.get(biz, 15)
        mid = eps * multiple
        conf = "Medium" if biz != "CYCLICAL" else "Low"

    low = mid * (1 - band / 100)
    high = mid * (1 + band / 100)
    return round(mid, 2), round(low, 2), round(high, 2), conf

def classify_business(ticker):
    t = ticker.upper()
    if t.endswith("BANK") or "BANK" in t:
        return "BANK"
    if t in ["INFY", "TCS", "HCLTECH"]:
        return "ASSET_LIGHT"
    return "GENERAL"

# ============================================================
# TEMPLATE
# ============================================================
def template():
    return pd.DataFrame([
        {"Ticker": "INFY", "Date": "2019-06-10", "Action": "BUY",  "Quantity": 10, "Price": 700,  "Charges": 0, "CMP": 0},
        {"Ticker": "INFY", "Date": "2020-01-15", "Action": "BUY",  "Quantity": 5,  "Price": 800,  "Charges": 0, "CMP": 0},
        {"Ticker": "INFY", "Date": "2021-05-20", "Action": "SELL", "Quantity": 3,  "Price": 1200, "Charges": 0, "CMP": 0},
        {"Ticker": "HDFCBANK", "Date": "2023-07-01", "Action": "BUY", "Quantity": 5, "Price": 1500, "Charges": 0, "CMP": 0},
    ])

with st.expander("📄 Download CSV Template"):
    st.dataframe(template(), hide_index=True)
    st.download_button(
        "Download Template",
        template().to_csv(index=False),
        file_name="PortfolioTemplate.csv"
    )

# ============================================================
# UPLOAD
# ============================================================
uploaded = st.file_uploader("Upload Portfolio CSV", type=["csv"])
if not uploaded:
    st.stop()

df = pd.read_csv(uploaded)
if any(c not in df.columns for c in REQUIRED_COLUMNS):
    st.error("CSV does not match required format.")
    st.stop()

df["Date"] = df["Date"].apply(parse_date)
df = df[df["Action"].isin(ACTION_ALLOWED)]
df["Quantity"] = pd.to_numeric(df["Quantity"])
df["Price"] = pd.to_numeric(df["Price"])
df["Charges"] = pd.to_numeric(df["Charges"])
df["CMP"] = pd.to_numeric(df["CMP"])

valuation_date = st.date_input("Valuation Date", value=date.today())

# ============================================================
# CMP EDIT
# ============================================================
cmp_df = df.groupby("Ticker")["CMP"].max().reset_index()
cmp_df = st.data_editor(cmp_df, hide_index=True)

cmp_map = dict(zip(cmp_df["Ticker"], cmp_df["CMP"]))

# ============================================================
# COMPUTE
# ============================================================
if not st.button("🚀 Compute"):
    st.stop()

rows = []
portfolio_cf = []

for tkr, g in df.groupby("Ticker"):
    qty = 0
    invested = 0
    realized = 0
    cf = []

    for _, r in g.iterrows():
        amt = r["Quantity"] * r["Price"]
        if r["Action"] == "BUY":
            amt = -(amt + r["Charges"])
            qty += r["Quantity"]
            invested += -amt
        else:
            amt = amt - r["Charges"]
            qty -= r["Quantity"]
            realized += amt

        cf.append((r["Date"], amt))
        portfolio_cf.append((r["Date"], amt))

    cmp = cmp_map.get(tkr, 0)
    value = qty * cmp
    if qty > 0:
        cf.append((valuation_date, value))

    r = xirr(cf)

    # Intrinsic
    biz = classify_business(tkr)
    eps = 50 if biz != "BANK" else 0   # Placeholder – override manually if needed
    bvps = 600 if biz == "BANK" else 0
    roe = 0.18 if biz == "BANK" else 0

    mid, lo, hi, conf = intrinsic_value(eps, bvps, roe, biz)

    mos = None
    if mid and cmp > 0:
        mos = round((mid - cmp) / mid * 100, 2)

    rows.append({
        "Ticker": tkr,
        "Qty": qty,
        "CMP": cmp,
        "Current Value": value,
        "XIRR %": None if r is None else round(r * 100, 2),
        "Intrinsic Mid": mid,
        "Intrinsic Range": f"{lo} – {hi}" if mid else "NA",
        "MoS %": mos,
        "Confidence": conf
    })

# ============================================================
# RESULTS
# ============================================================
result_df = pd.DataFrame(rows)
st.subheader("📊 Stock-wise XIRR + Intrinsic")
st.dataframe(result_df, hide_index=True)

portfolio_cf.append((valuation_date, result_df["Current Value"].sum()))
pxirr = xirr(portfolio_cf)

# ============================================================
# NIFTY BENCHMARK
# ============================================================
start = min(d for d, _ in portfolio_cf)
nifty = yf.download("^NSEI", start=start, end=valuation_date)
prices = nifty["Close"].dropna()

units = 0
bench_cf = []

for d, amt in portfolio_cf:
    px = prices[prices.index <= pd.Timestamp(d)].iloc[-1]
    if amt < 0:
        units += abs(amt) / px
        bench_cf.append((d, amt))

bench_cf.append((valuation_date, units * prices.iloc[-1]))
nxirr = xirr(bench_cf)

st.subheader("📈 Portfolio vs NIFTY")
c1, c2, c3 = st.columns(3)
c1.metric("Portfolio XIRR", f"{pxirr*100:.2f}%" if pxirr else "NA")
c2.metric("NIFTY XIRR", f"{nxirr*100:.2f}%" if nxirr else "NA")
c3.metric("Alpha", f"{(pxirr-nxirr)*100:.2f}%" if pxirr and nxirr else "NA")

st.download_button(
    "⬇️ Download Results",
    result_df.to_csv(index=False),
    file_name="xirr_intrinsic_results.csv"
)

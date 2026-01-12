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

st.set_page_config(page_title="Portfolio XIRR & Intrinsic Analyzer", layout="wide")
st.title("📈 Portfolio XIRR + Intrinsic Valuation + NIFTY Benchmark")

st.warning(
    "⚠️ Corporate actions (splits/bonus) and dividends are NOT auto-captured. "
    "If quantities are not adjusted manually, XIRR & intrinsic values may be skewed."
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
    if not (any(a < 0 for _, a in cashflows) and any(a > 0 for _, a in cashflows)):
        return None

    r = 0.15
    for _ in range(60):
        f = xnpv(r, cashflows)
        df = (xnpv(r + 1e-5, cashflows) - f) / 1e-5
        if abs(df) < 1e-10:
            break
        r -= f / df
        r = max(-0.99, min(r, 10))
    return r

# ============================================================
# NIFTY BENCHMARK (ROBUST)
# ============================================================

def compute_nifty_xirr(portfolio_cf, valuation_date):
    start = min(d for d, _ in portfolio_cf)
    df = yf.download("^NSEI", start=start - timedelta(days=10),
                     end=valuation_date + timedelta(days=10), progress=False)
    if df.empty:
        return None, "NIFTY data unavailable"

    prices = df["Close"].dropna()
    units = 0.0
    bench_cf = []

    for d, amt in portfolio_cf:
        if amt < 0:
            px = prices[prices.index <= pd.Timestamp(d)].iloc[-1]
            units += abs(amt) / px
            bench_cf.append((d, amt))

    final_px = prices.iloc[-1]
    bench_cf.append((valuation_date, units * final_px))
    return xirr(bench_cf), "Used last available close"

# ============================================================
# FUNDAMENTALS (CACHED, RATE-LIMIT SAFE)
# ============================================================

@st.cache_data(ttl=86400)
def get_stock_fundamentals(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        fin = t.financials

        eps_series = None
        shares = info.get("sharesOutstanding")
        if fin is not None and not fin.empty and shares:
            if "Net Income" in fin.index:
                eps_series = (fin.loc["Net Income"] / shares).dropna()

        return {
            "sector": info.get("sector", ""),
            "roe": info.get("returnOnEquity"),
            "bvps": info.get("bookValue"),
            "trailing_eps": info.get("trailingEps"),
            "eps_series": eps_series
        }
    except:
        return {}

# ============================================================
# INTRINSIC VALUATION (FINAL DESIGN)
# ============================================================

def classify_business(sector):
    s = sector.lower()
    if "bank" in s:
        return "BANK"
    if any(x in s for x in ["it", "software", "pharma"]):
        return "ASSET_LIGHT"
    if any(x in s for x in ["metal", "energy", "oil", "power"]):
        return "CYCLICAL"
    return "GENERAL"

def intrinsic_for_stock(ticker, cmp):
    data = get_stock_fundamentals(ticker)
    if not data:
        return None, None, "Low"

    biz = classify_business(data["sector"])
    eps_series = data["eps_series"]
    trailing_eps = data["trailing_eps"]
    roe = data["roe"]
    bvps = data["bvps"]

    # ---- BANKS ----
    if biz == "BANK":
        if bvps and roe:
            target_pb = max(0.8, min(2.0, roe / 0.13))
            floor = bvps * target_pb * 0.9
            ceiling = floor * 1.20
            return round(floor,2), round(ceiling,2), "High"
        return None, None, "Low"

    # ---- EPS NORMALIZATION ----
    norm_eps = None
    confidence = "Low"

    if eps_series is not None and len(eps_series) >= 3:
        norm_eps = eps_series.tail(3).mean()
        confidence = "High"
    elif trailing_eps:
        norm_eps = trailing_eps * 0.8  # conservative fallback
        confidence = "Medium"

    if not norm_eps or norm_eps <= 0:
        return None, None, "Low"

    # ---- VALUATION BY TYPE ----
    if biz == "ASSET_LIGHT":
        floor, ceiling = norm_eps * 16, norm_eps * 22
    elif biz == "CYCLICAL":
        floor, ceiling = norm_eps * 9, norm_eps * 13
        confidence = "Low"
    else:  # GENERAL
        floor, ceiling = norm_eps * 13, norm_eps * 18
        if confidence == "High":
            confidence = "Medium"

    return round(floor,2), round(ceiling,2), confidence

def margin_of_safety(cmp, floor):
    if cmp and floor:
        return round((floor - cmp) / floor * 100, 2)
    return None

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
            floor, ceiling, conf = intrinsic_for_stock(tkr, cmp)
            mos = margin_of_safety(cmp, floor)

            rows.append({
                "Ticker": tkr,
                "Holding Qty": qty,
                "CMP": cmp,
                "Current Value": curr_val,
                "Total Invested": invested,
                "Total Realized": realized,
                "XIRR %": None if r is None else round(r * 100, 2),
                "Intrinsic Range (₹)": f"{floor} – {ceiling}" if floor else "NA",
                "MoS %": mos,
                "Valuation Confidence": conf
            })

        stock_df = pd.DataFrame(rows)
        st.subheader("📌 Stock-wise XIRR + Intrinsic")
        st.dataframe(stock_df, use_container_width=True)

        px = xirr(portfolio_cf)
        nx, note = compute_nifty_xirr(portfolio_cf, valuation_date)

        st.subheader("📊 Portfolio vs NIFTY 50")
        c1, c2, c3 = st.columns(3)
        c1.metric("Portfolio XIRR", f"{px*100:.2f}%" if px else "NA")
        c2.metric("NIFTY XIRR", f"{nx*100:.2f}%" if nx else "NA")
        c3.metric("Alpha", f"{(px-nx)*100:.2f}%" if px and nx else "NA")

        st.caption(f"NIFTY benchmark note: {note}")

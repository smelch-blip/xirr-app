Here is a refined, ready-to-use version of your app code with:

- Safer yfinance usage (cached, no EPS-series gymnastics).
- Cleaner intrinsic logic using trailing/normalized EPS.
- Better null checks and display logic.

```python
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
        # Newton iteration
        try:
            r -= f / df
        except ZeroDivisionError:
            return None
        r = max(-0.99, min(r, 10))
    return r

# ============================================================
# NIFTY BENCHMARK (ROBUST)
# ============================================================

@st.cache_data(ttl=86400)
def get_nifty_history(start, end):
    df = yf.download("^NSEI", start=start, end=end, progress=False)
    return df

def compute_nifty_xirr(portfolio_cf, valuation_date):
    if not portfolio_cf:
        return None, "No portfolio cashflows"

    start = min(d for d, _ in portfolio_cf)
    df = get_nifty_history(start - timedelta(days=10), valuation_date + timedelta(days=10))
    if df is None or df.empty:
        return None, "NIFTY data unavailable"

    prices = df["Close"].dropna()
    if prices.empty:
        return None, "NIFTY prices unavailable"

    units = 0.0
    bench_cf = []

    for d, amt in portfolio_cf:
        if amt < 0:
            closest = prices[prices.index <= pd.Timestamp(d)]
            if closest.empty:
                continue
            px = closest.iloc[-1]
            units += abs(amt) / px
            bench_cf.append((d, amt))

    if units <= 0:
        return None, "No buys to simulate benchmark"

    final_px = prices.iloc[-1]
    bench_cf.append((valuation_date, units * final_px))
    return xirr(bench_cf), "Used last available close"

# ============================================================
# FUNDAMENTALS (CACHED, RATE-LIMIT SAFE)
# ============================================================

@st.cache_data(ttl=86400)
def get_stock_fundamentals(ticker):
    """
    Single yfinance call per ticker, cached.
    Uses trailing/forward EPS and simple normalization.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        trailing_eps = info.get("trailingEps") or 0
        forward_eps = info.get("forwardEps") or 0

        # Conservative normalized EPS:
        if trailing_eps and forward_eps:
            norm_eps = (trailing_eps + forward_eps) / 2
        elif trailing_eps:
            norm_eps = trailing_eps * 0.9
        elif forward_eps:
            norm_eps = forward_eps * 0.8
        else:
            norm_eps = 0

        return {
            "sector": (info.get("sector") or "").lower(),
            "roe": info.get("returnOnEquity") or 0,
            "bvps": info.get("bookValue") or 0,
            "trailing_eps": float(trailing_eps) if trailing_eps else 0.0,
            "norm_eps": float(norm_eps) if norm_eps else 0.0,
        }
    except Exception:
        return {
            "sector": "",
            "roe": 0,
            "bvps": 0,
            "trailing_eps": 0.0,
            "norm_eps": 0.0,
        }

# ============================================================
# INTRINSIC VALUATION (FINAL DESIGN)
# ============================================================

def classify_business(sector: str):
    s = (sector or "").lower()
    if "bank" in s:
        return "BANK"
    if any(x in s for x in ["it", "software", "pharma", "services"]):
        return "ASSET_LIGHT"
    if any(x in s for x in ["metal", "energy", "oil", "power", "commodity"]):
        return "CYCLICAL"
    return "GENERAL"

def intrinsic_for_stock(ticker, cmp_price):
    data = get_stock_fundamentals(ticker)
    sector = data["sector"]
    roe = data["roe"]
    bvps = data["bvps"]
    norm_eps = data["norm_eps"]
    trailing_eps = data["trailing_eps"]

    biz = classify_business(sector)

    # ---- BANKS ----
    if biz == "BANK":
        if bvps > 0 and roe:
            # Cost of equity ~ 13% for Indian banks, clipped to reasonable bounds
            target_pb = max(0.8, min(2.0, roe / 0.13))
            floor = bvps * target_pb * 0.9     # small MoS inside band
            ceiling = floor * 1.20
            return round(floor, 2), round(ceiling, 2), "High"
        return None, None, "Low"

    # ---- EPS NORMALIZATION ----
    eps = norm_eps if norm_eps > 0 else trailing_eps * 0.8
    if eps <= 0:
        return None, None, "Low"

    confidence = "Medium"

    # ---- VALUATION BY TYPE ----
    if biz == "ASSET_LIGHT":
        # Quality compounders: 16–22x normalized EPS
        floor, ceiling = eps * 16, eps * 22
        confidence = "High"
    elif biz == "CYCLICAL":
        # Cyclicals: tighter mid-cycle band with low confidence
        floor, ceiling = eps * 9, eps * 13
        confidence = "Low"
    else:  # GENERAL
        floor, ceiling = eps * 13, eps * 18
        # Do not upgrade to High; keep Medium at best
        confidence = "Medium"

    return round(floor, 2), round(ceiling, 2), confidence

def margin_of_safety(cmp_price, floor):
    if cmp_price is None or floor is None:
        return None
    if cmp_price <= 0 or floor <= 0:
        return None
    return round((floor - cmp_price) / floor * 100, 2)

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

    # Allow user to tweak CMPs
    cmp_df = df.groupby("Ticker")["CMP"].max().reset_index()
    cmp_edit = st.data_editor(cmp_df, hide_index=True)

    if st.button("🚀 Compute XIRR & Intrinsic"):
        cmp_map = dict(zip(cmp_edit["Ticker"], cmp_edit["CMP"]))
        rows = []
        portfolio_cf = []

        for tkr, g in df.groupby("Ticker"):
            qty = 0
            invested = 0.0
            realized = 0.0
            cashflows = []

            for _, r in g.iterrows():
                if r["Action"] == "BUY":
                    amt = -(r["Quantity"] * r["Price"] + r["Charges"])
                    qty += r["Quantity"]
                    invested += -amt
                else:  # SELL
                    amt = r["Quantity"] * r["Price"] - r["Charges"]
                    qty -= r["Quantity"]
                    realized += amt

                cashflows.append((r["Date"], amt))
                portfolio_cf.append((r["Date"], amt))

            cmp_price = float(cmp_map.get(tkr, 0))
            curr_val = qty * cmp_price
            if qty > 0:
                cashflows.append((valuation_date, curr_val))
                portfolio_cf.append((valuation_date, curr_val))

            r = xirr(cashflows)
            floor, ceiling, conf = intrinsic_for_stock(tkr, cmp_price)
            mos = margin_of_safety(cmp_price, floor)

            if floor is not None and ceiling is not None:
                intr_str = f"{floor} – {ceiling}"
            else:
                intr_str = "NA"

            rows.append({
                "Ticker": tkr,
                "Holding Qty": qty,
                "CMP": cmp_price,
                "Current Value": curr_val,
                "Total Invested": invested,
                "Total Realized": realized,
                "XIRR %": None if r is None else round(r * 100, 2),
                "Intrinsic Range (₹)": intr_str,
                "MoS %": mos,
                "Valuation Confidence": conf,
            })

        stock_df = pd.DataFrame(rows)
        st.subheader("📌 Stock-wise XIRR + Intrinsic")
        st.dataframe(stock_df, use_container_width=True)

        px = xirr(portfolio_cf)
        nx, note = compute_nifty_xirr(portfolio_cf, valuation_date)

        st.subheader("📊 Portfolio vs NIFTY 50")
        c1, c2, c3 = st.columns(3)
        c1.metric("Portfolio XIRR", f"{px*100:.2f}%" if px is not None else "NA")
        c2.metric("NIFTY XIRR", f"{nx*100:.2f}%" if nx is not None else "NA")
        c3.metric(
            "Alpha",
            f"{(px - nx)*100:.2f}%" if (px is not None and nx is not None) else "NA",
        )

        st.caption(f"NIFTY benchmark note: {note}")
```

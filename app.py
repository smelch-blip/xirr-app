import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date

# ============================================================
# Helpers: XIRR (robust bisection with sign-change search)
# ============================================================
def _to_datetime(x):
    # Accepts many formats (YYYY-MM-DD, DD/MM/YYYY, etc.)
    return pd.to_datetime(x, errors="coerce").to_pydatetime()

def xnpv(rate, cashflows):
    """
    cashflows: list of (datetime, amount)
    """
    if rate <= -0.999999:
        return np.nan
    t0 = cashflows[0][0]
    total = 0.0
    for t, cf in cashflows:
        days = (t - t0).days
        total += cf / ((1 + rate) ** (days / 365.0))
    return total

def xirr(cashflows):
    """
    Returns annualized IRR as a decimal (e.g. 0.18 = 18%),
    or None if cannot be solved reliably.
    """
    cashflows = sorted(cashflows, key=lambda x: x[0])
    amounts = [cf for _, cf in cashflows]

    # Need at least one negative and one positive cashflow
    if not (any(a < 0 for a in amounts) and any(a > 0 for a in amounts)):
        return None

    # Try to find a bracket [lo, hi] where NPV changes sign
    def f(r): 
        v = xnpv(r, cashflows)
        return v

    # Grid search for sign change (handles weird cashflow shapes better)
    grid = np.concatenate([
        np.linspace(-0.90, -0.10, 9),
        np.linspace(-0.10, 0.10, 9),
        np.linspace(0.10, 1.00, 10),
        np.linspace(1.00, 5.00, 9),
        np.array([10.0])  # 1000% upper cap
    ])

    vals = []
    for r in grid:
        try:
            vals.append(f(r))
        except Exception:
            vals.append(np.nan)

    bracket = None
    for i in range(len(grid) - 1):
        a, b = vals[i], vals[i+1]
        if np.isfinite(a) and np.isfinite(b) and (a == 0 or b == 0 or (a * b < 0)):
            bracket = (grid[i], grid[i+1])
            break

    if bracket is None:
        return None

    lo, hi = bracket
    flo, fhi = f(lo), f(hi)
    if not (np.isfinite(flo) and np.isfinite(fhi)):
        return None

    # Bisection
    for _ in range(80):
        mid = (lo + hi) / 2
        fmid = f(mid)
        if not np.isfinite(fmid):
            return None
        if abs(fmid) < 1e-6:
            return mid
        if flo * fmid < 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid

    return (lo + hi) / 2

# ============================================================
# Price fetch (Yahoo)
# ============================================================
@st.cache_data(ttl=1800)
def fetch_last_price(ticker: str):
    """
    Returns last close price using yfinance.
    """
    try:
        # Using 5d to be resilient
        df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            return None
        return float(df["Close"].dropna().iloc[-1])
    except Exception:
        return None

def normalize_ticker(ticker: str, exchange: str):
    t = str(ticker).strip().upper()
    ex = (exchange or "").strip().upper()
    # If user already provided suffix, keep it.
    if "." in t:
        return t
    if ex == "NSE":
        return t + ".NS"
    if ex == "BSE":
        return t + ".BO"
    # Default to NSE if not specified
    return t + ".NS"

# ============================================================
# App
# ============================================================
st.set_page_config(page_title="Stock-wise XIRR Analyzer", layout="wide")
st.title("📈 Stock-wise XIRR Analyzer (Upload → Stock XIRR + Portfolio XIRR)")

st.markdown("""
Upload your **transactions** (BUY/SELL/DIVIDEND/FEES).  
The app computes:
- **Stock-wise XIRR** (money-weighted return)
- **First Buy Date** + holding age (so comparisons are fair)
- **Overall Portfolio XIRR**
""")

with st.expander("✅ Required upload format (CSV/Excel)"):
    st.code(
        "Date,Ticker,Type,Quantity,Price,Exchange,Amount,Notes\n"
        "2021-06-15,TCS,BUY,10,3200,NSE,,First buy\n"
        "2022-01-10,TCS,DIVIDEND,0,0,NSE,450,Dividend received\n"
        "2023-09-05,TCS,SELL,2,3800,NSE,,Partial sell\n"
    )

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

as_of = st.date_input("As-of date (valuation date for open holdings)", value=date.today())
use_yahoo_prices = st.checkbox("Fetch latest price from Yahoo for open holdings (recommended)", value=True)

# Optional: allow user-provided price overrides (useful if Yahoo fails)
st.caption("Tip: If Yahoo fails for some microcaps, you can add an `LTP` column per ticker later (enhancement), or we can extend this app to pull from NSE APIs.")

if uploaded:
    # Read file
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    # Normalize columns
    df.columns = [c.strip() for c in df.columns]

    required = ["Date", "Ticker", "Type", "Quantity", "Price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    if "Exchange" not in df.columns:
        df["Exchange"] = "NSE"

    if "Amount" not in df.columns:
        df["Amount"] = np.nan

    # Parse/clean
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", "Type"]).copy()

    df["Type"] = df["Type"].astype(str).str.strip().str.upper()
    valid_types = {"BUY", "SELL", "DIVIDEND", "FEES"}
    df = df[df["Type"].isin(valid_types)].copy()

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0.0)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    # Normalize tickers
    df["NormTicker"] = df.apply(lambda r: normalize_ticker(r["Ticker"], r.get("Exchange", "NSE")), axis=1)

    # Build cashflows
    def row_cashflow(r):
        amt = r["Amount"]
        if pd.isna(amt):
            amt = r["Quantity"] * r["Price"]
        typ = r["Type"]
        if typ == "BUY":
            return -abs(float(amt))
        if typ in ("SELL", "DIVIDEND"):
            return abs(float(amt))
        if typ == "FEES":
            return -abs(float(amt))
        return 0.0

    df["Cashflow"] = df.apply(row_cashflow, axis=1)

    # Summary per ticker
    tickers = sorted(df["NormTicker"].unique().tolist())

    price_map = {}
    if use_yahoo_prices:
        with st.spinner("Fetching latest prices for open holdings..."):
            for t in tickers:
                price_map[t] = fetch_last_price(t)

    rows = []
    portfolio_cashflows = []

    as_of_dt = datetime(as_of.year, as_of.month, as_of.day)

    for t in tickers:
        dft = df[df["NormTicker"] == t].copy()
        dft = dft.sort_values("Date")

        # Net quantity (buys - sells)
        buys_qty = dft.loc[dft["Type"] == "BUY", "Quantity"].sum()
        sells_qty = dft.loc[dft["Type"] == "SELL", "Quantity"].sum()
        net_qty = float(buys_qty - sells_qty)

        first_buy = dft.loc[dft["Type"] == "BUY", "Date"].min()
        last_txn = dft["Date"].max()

        invested = -dft.loc[dft["Cashflow"] < 0, "Cashflow"].sum()   # positive number
        inflows  = dft.loc[dft["Cashflow"] > 0, "Cashflow"].sum()

        # Cashflows list for XIRR
        cfs = [(dt.to_pydatetime(), float(cf)) for dt, cf in zip(dft["Date"], dft["Cashflow"])]

        # Add terminal value for open holdings
        ltp = price_map.get(t) if use_yahoo_prices else None
        current_value = None
        terminal_added = False

        if net_qty != 0:
            if ltp is not None:
                current_value = net_qty * ltp
                cfs.append((as_of_dt, float(current_value)))
                terminal_added = True

        # Compute stock XIRR
        r = xirr(cfs)
        r_pct = None if r is None else (r * 100.0)

        # Total P&L view (simple): inflows + current_value - invested
        pnl = None
        if current_value is not None:
            pnl = inflows + current_value - invested
        else:
            pnl = inflows - invested  # closed positions or no price

        # Add to portfolio cashflows:
        portfolio_cashflows.extend(cfs if terminal_added else [(dt, cf) for dt, cf in cfs])

        rows.append({
            "Ticker": t,
            "First Buy Date": None if pd.isna(first_buy) else first_buy.date(),
            "Last Transaction Date": last_txn.date(),
            "Net Quantity": round(net_qty, 4),
            "Invested (₹)": round(float(invested), 2),
            "Inflows (Sell+Div) (₹)": round(float(inflows), 2),
            "LTP Used (₹)": None if ltp is None else round(float(ltp), 2),
            "Current Value (₹)": None if current_value is None else round(float(current_value), 2),
            "P&L (₹)": None if pnl is None else round(float(pnl), 2),
            "XIRR %": None if r_pct is None else round(float(r_pct), 2),
            "Notes": "Price missing → XIRR may be N/A for open holdings" if (net_qty != 0 and ltp is None) else ""
        })

    res = pd.DataFrame(rows)

    st.subheader("Stock-wise XIRR (with holding context)")
    st.dataframe(res, use_container_width=True, hide_index=True)

    # Portfolio XIRR (aggregate)
    # De-duplicate exact same terminal cashflow duplicates risk: keep as is because per-stock adds are correct.
    pr = xirr(sorted(portfolio_cashflows, key=lambda x: x[0]))
    if pr is not None:
        st.subheader("Overall Portfolio XIRR")
        st.metric("Portfolio XIRR %", f"{pr*100:.2f}%")
    else:
        st.subheader("Overall Portfolio XIRR")
        st.warning("Could not compute portfolio XIRR (needs at least one negative and one positive cashflow and a solvable rate).")

    with st.expander("How to interpret this correctly (important)"):
        st.markdown("""
- **XIRR is money-weighted.** It depends on when you added money and when you exited/valued.
- To compare stocks fairly, look at **First Buy Date** and holding age alongside **XIRR**.
- For open holdings, XIRR depends on the **as-of price** (we add a terminal value cashflow).
- Very short holding periods can show **wild XIRR**. That’s normal.
""")

# ============================================================
# Footer: run instructions
# ============================================================
st.markdown("---")
st.markdown("### Run locally")
st.code(
    "pip install -r requirements.txt\n"
    "streamlit run app.py"
)

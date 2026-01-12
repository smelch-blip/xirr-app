import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

# =========================
# CONFIG
# =========================
REQUIRED_COLUMNS = ["Ticker", "Date", "Action", "Quantity", "Price", "Charges", "CMP"]
ACTION_ALLOWED = {"BUY", "SELL"}  # corporate actions/dividends not handled in this version

st.set_page_config(page_title="Stock XIRR Analyzer", layout="wide")
st.title("📈 Stock-wise XIRR + Portfolio XIRR (Upload → CMP Editable → Results)")

st.warning(
    "⚠️ Corporate actions (splits/bonus) and dividends are NOT auto-captured in this version. "
    "If these apply to your holdings, XIRR can be skewed."
)

# =========================
# HELPERS
# =========================
def parse_date_safe(x):
    if pd.isna(x):
        return None
    if isinstance(x, (datetime, date)):
        return pd.to_datetime(x).date()
    try:
        return pd.to_datetime(str(x), dayfirst=True, errors="coerce").date()
    except Exception:
        return None

def to_float_safe(x, default=0.0):
    try:
        if pd.isna(x) or x == "":
            return default
        return float(x)
    except Exception:
        return default

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
    """
    Returns rate as decimal (0.15 = 15%), or None if not solvable.
    Uses Newton then bisection fallback.
    """
    if len(cashflows) < 2:
        return None
    amounts = [amt for _, amt in cashflows]
    if not (any(a < 0 for a in amounts) and any(a > 0 for a in amounts)):
        return None

    # Newton-Raphson
    r = 0.15
    for _ in range(60):
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

    # Bracket + Bisection
    grid = np.concatenate([
        np.linspace(-0.9, 0.0, 30),
        np.linspace(0.0, 1.0, 50),
        np.linspace(1.0, 10.0, 50),
    ])
    prev_r = grid[0]
    prev_f = xnpv(prev_r, cashflows)
    bracket = None
    for rr in grid[1:]:
        ff = xnpv(rr, cashflows)
        if np.isfinite(prev_f) and np.isfinite(ff) and (prev_f == 0 or ff == 0 or (prev_f * ff < 0)):
            bracket = (prev_r, rr)
            break
        prev_r, prev_f = rr, ff

    if bracket is None:
        return None

    a, b = bracket
    fa, fb = xnpv(a, cashflows), xnpv(b, cashflows)
    if not (np.isfinite(fa) and np.isfinite(fb)):
        return None

    for _ in range(80):
        mid = (a + b) / 2.0
        fm = xnpv(mid, cashflows)
        if not np.isfinite(fm):
            return None
        if abs(fm) < 1e-8:
            return mid
        if fa * fm < 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return (a + b) / 2.0
# =========================
# BENCHMARK: NIFTY 50 XIRR
# =========================
import yfinance as yf
...
def compute_nifty_xirr(...):
    ...
def make_template():
    return pd.DataFrame([
        {"Ticker": "INFY", "Date": "2019-06-10", "Action": "BUY",  "Quantity": 10, "Price": 700,  "Charges": 0, "CMP": 0},
        {"Ticker": "INFY", "Date": "2020-01-15", "Action": "BUY",  "Quantity": 5,  "Price": 800,  "Charges": 0, "CMP": 0},
        {"Ticker": "INFY", "Date": "2021-05-20", "Action": "SELL", "Quantity": 3,  "Price": 1200, "Charges": 0, "CMP": 0},
        {"Ticker": "TCS",  "Date": "2023-07-01", "Action": "BUY",  "Quantity": 2,  "Price": 3500, "Charges": 0, "CMP": 0},
    ])

def validate_columns(df):
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]

def clean_transactions(df):
    df = df.copy()[REQUIRED_COLUMNS]
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Action"] = df["Action"].astype(str).str.strip().str.upper()
    df["Date"] = df["Date"].apply(parse_date_safe)
    df = df[df["Date"].notna()].copy()

    for col in ["Quantity", "Price", "Charges", "CMP"]:
        df[col] = df[col].apply(to_float_safe)

    unsupported = df[~df["Action"].isin(ACTION_ALLOWED)]
    if len(unsupported) > 0:
        st.info(
            f"ℹ️ Ignoring {len(unsupported)} rows with unsupported Action "
            f"(only BUY/SELL used): {sorted(set(unsupported['Action'].tolist()))}"
        )
        df = df[df["Action"].isin(ACTION_ALLOWED)].copy()

    df = df[df["Ticker"].ne("")].copy()
    df = df[df["Quantity"] > 0].copy()
    df = df[df["Price"] >= 0].copy()
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df

def build_cmp_table(df):
    cmp_map = {}
    for tkr, g in df.groupby("Ticker"):
        vals = g["CMP"].tolist()
        nz = [v for v in vals if v and v > 0]
        cmp_map[tkr] = float(nz[-1]) if nz else (float(vals[-1]) if vals else 0.0)
    return pd.DataFrame([{"Ticker": k, "CMP": v} for k, v in sorted(cmp_map.items())])

def compute_xirr(df, cmp_override, valuation_date):
    cmp_override = cmp_override.set_index("Ticker")["CMP"].to_dict()

    rows = []
    all_cashflows = []

    for tkr, g in df.groupby("Ticker"):
        g = g.sort_values("Date").copy()
        first_buy = g[g["Action"] == "BUY"]["Date"].min()
        last_txn = g["Date"].max()

        qty = 0.0
        invested = 0.0
        realized = 0.0
        cashflows = []

        for _, r in g.iterrows():
            q = float(r["Quantity"])
            p = float(r["Price"])
            ch = float(r["Charges"])
            dt = r["Date"]

            if r["Action"] == "BUY":
                amt = -(q * p + ch)
                qty += q
                invested += (q * p + ch)
            else:
                amt = +(q * p - ch)
                qty -= q
                realized += (q * p - ch)

            cashflows.append((dt, amt))

        cmp_val = float(cmp_override.get(tkr, 0.0))
        current_value = qty * cmp_val

        # terminal valuation cashflow
        if abs(current_value) > 1e-9:
            cashflows.append((valuation_date, current_value))

        r = xirr(cashflows)
        all_cashflows.extend(cashflows)

        rows.append({
            "Ticker": tkr,
            "First Buy Date": first_buy,
            "Last Transaction Date": last_txn,
            "Holding Qty (Current)": round(qty, 4),
            "CMP (Editable)": round(cmp_val, 2),
            "Current Value (Qty×CMP)": round(current_value, 2),
            "Total Invested (Buys incl Charges)": round(invested, 2),
            "Total Realized (Sells net Charges)": round(realized, 2),
            "XIRR %": None if r is None else round(r * 100, 2),
            "XIRR Status": "Not computable" if r is None else "OK",
        })

    stock_df = pd.DataFrame(rows).sort_values(["XIRR %"], ascending=False, na_position="last")

    overall_r = xirr(all_cashflows)
    overall = {
        "Overall XIRR %": None if overall_r is None else round(overall_r * 100, 2),
        "Total Invested": round(stock_df["Total Invested (Buys incl Charges)"].sum(), 2) if not stock_df.empty else 0.0,
        "Total Realized": round(stock_df["Total Realized (Sells net Charges)"].sum(), 2) if not stock_df.empty else 0.0,
        "Total Current Value": round(stock_df["Current Value (Qty×CMP)"].sum(), 2) if not stock_df.empty else 0.0,
        "Valuation Date": valuation_date,
    }
    return stock_df, overall

# =========================
# UI: TEMPLATE
# =========================
with st.expander("📄 Download the exact upload format (template)"):
    tmpl = make_template()
    st.dataframe(tmpl, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download template CSV",
        data=tmpl.to_csv(index=False).encode("utf-8"),
        file_name="PortfolioImportTemplate.csv",
        mime="text/csv"
    )
    st.markdown("**Exact column names required:** `Ticker, Date, Action, Quantity, Price, Charges, CMP`")

# =========================
# UI: UPLOAD
# =========================
st.markdown("### Upload your transactions CSV")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    raw = pd.read_csv(uploaded)
    missing = validate_columns(raw)
    if missing:
        st.error(f"Missing columns: {missing}. Please use the template and keep EXACT names.")
        st.stop()

    df = clean_transactions(raw)
    if df.empty:
        st.error("No valid BUY/SELL rows after cleaning. Check your file.")
        st.stop()

    st.markdown("### Choose valuation date (usually today)")
    valuation_date = st.date_input("Valuation date", value=date.today())

    st.markdown("### Edit CMP per stock (because Yahoo can be flaky)")
    cmp_df = build_cmp_table(df)
    edited_cmp = st.data_editor(
        cmp_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "Ticker": st.column_config.TextColumn(disabled=True),
            "CMP": st.column_config.NumberColumn(min_value=0.0, step=0.05),
        }
    )

    if st.button("🚀 Compute Stock XIRR + Overall XIRR"):
        stock_df, overall = compute_xirr(df, edited_cmp, valuation_date)

        st.subheader("✅ Overall Portfolio Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall XIRR %", "—" if overall["Overall XIRR %"] is None else f'{overall["Overall XIRR %"]}%')
        c2.metric("Total Invested", f'₹{overall["Total Invested"]:,.0f}')
        c3.metric("Total Realized", f'₹{overall["Total Realized"]:,.0f}')
        c4.metric("Total Current Value", f'₹{overall["Total Current Value"]:,.0f}')

        st.subheader("📌 Stock-wise XIRR (sorted by XIRR)")
        st.dataframe(stock_df, use_container_width=True, hide_index=True)
        st.divider()
st.subheader("📊 Benchmark vs NIFTY 50")

        st.download_button(
            "⬇️ Download results CSV",
            data=stock_df.to_csv(index=False).encode("utf-8"),
            file_name="stock_xirr_results.csv",
            mime="text/csv"
        )
st.divider()
st.subheader("📊 Benchmark vs NIFTY 50")
        
        st.info(
            "XIRR = BUY/SELL cashflows + terminal valuation cashflow (Qty×CMP) on the valuation date. "
            "If CMP is wrong or corporate actions/dividends matter, XIRR will be skewed."
        )

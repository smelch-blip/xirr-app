import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ============================================================
# CONFIG
# ============================================================
REQUIRED_COLUMNS = ["Ticker", "Date", "Action", "Quantity", "Price", "Charges", "CMP"]
ACTION_ALLOWED = {"BUY", "SELL"}

st.set_page_config(page_title="Portfolio XIRR + Intrinsic + NIFTY", layout="wide")
st.title("📈 Portfolio XIRR + Intrinsic Valuation + NIFTY 50 Benchmark")

st.warning(
    "⚠️ Corporate actions (splits/bonus) and dividends are NOT auto-captured in this version. "
    "If your quantities/prices are not adjusted manually, XIRR & intrinsic values may be skewed."
)

# ============================================================
# SIDEBAR SETTINGS
# ============================================================
st.sidebar.header("Settings")

default_market = st.sidebar.selectbox(
    "Default market suffix for tickers",
    ["NSE (.NS)", "BSE (.BO)", "None"],
    index=0
)

use_autofetch = st.sidebar.toggle(
    "Auto-fetch fundamentals from Yahoo (yfinance)",
    value=True,
    help="If Yahoo rate-limits or misses India fundamentals, use the overrides table."
)

use_dcf = st.sidebar.toggle(
    "Compute DCF intrinsic (non-financials only)",
    value=True,
    help="DCF requires FCF/OCF + Shares (and ideally cash/debt). For Banks/NBFC it is disabled automatically."
)

sensitivity_pct = st.sidebar.slider(
    "Intrinsic range sensitivity (±%) around midpoint",
    min_value=5, max_value=20, value=12, step=1,
    help="Keeps intrinsic range realistic (not huge). Example: ±12% around midpoint."
)

sleep_between_calls = st.sidebar.slider(
    "Throttle Yahoo calls (seconds between tickers)",
    min_value=0.0, max_value=1.0, value=0.2, step=0.05,
    help="Higher reduces rate-limit risk on Streamlit Cloud."
)

st.sidebar.caption(
    "Tip: If you see `YFRateLimitError`, turn off Auto-fetch and fill overrides for EPS/BVPS/ROE/FCF/Shares."
)

# ============================================================
# HELPERS: DATE + CLEANING
# ============================================================
def parse_date_safe(x):
    if pd.isna(x):
        return None
    if isinstance(x, (datetime, date)):
        return pd.to_datetime(x).date()
    try:
        d = pd.to_datetime(str(x), errors="coerce")
        if pd.isna(d):
            return None
        return d.date()
    except Exception:
        return None

def to_float_safe(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def validate_columns(df):
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()[REQUIRED_COLUMNS]
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Action"] = df["Action"].astype(str).str.strip().str.upper()
    df["Date"] = df["Date"].apply(parse_date_safe)
    df = df[df["Date"].notna()].copy()

    for col in ["Quantity", "Price", "Charges", "CMP"]:
        df[col] = df[col].apply(lambda v: to_float_safe(v, np.nan))

    df = df[df["Action"].isin(ACTION_ALLOWED)].copy()
    df = df[df["Ticker"].ne("")].copy()
    df = df[df["Quantity"].notna() & (df["Quantity"] > 0)].copy()
    df = df[df["Price"].notna() & (df["Price"] >= 0)].copy()
    df["Charges"] = df["Charges"].fillna(0.0)
    df["CMP"] = df["CMP"].fillna(0.0)

    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df

def build_cmp_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tkr, g in df.groupby("Ticker"):
        vals = [to_float_safe(v, np.nan) for v in g["CMP"].tolist()]
        nz = [v for v in vals if np.isfinite(v) and v > 0]
        cmp_val = nz[-1] if nz else (vals[-1] if vals else 0.0)
        if not np.isfinite(cmp_val):
            cmp_val = 0.0
        rows.append({"Ticker": tkr, "CMP": float(cmp_val)})
    return pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)

# ============================================================
# XIRR ENGINE (Newton + Bracket/Bisection fallback)
# ============================================================
def xnpv(rate, cashflows):
    if rate <= -0.999999:
        return np.inf
    t0 = min(dt for dt, _ in cashflows)
    total = 0.0
    for dt, amt in cashflows:
        years = (dt - t0).days / 365.25
        total += float(amt) / ((1.0 + rate) ** years)
    return total

def xirr(cashflows):
    if not cashflows or len(cashflows) < 2:
        return None

    cf = [(d, float(a)) for d, a in cashflows if d is not None and a is not None]
    if len(cf) < 2:
        return None

    amounts = [a for _, a in cf]
    if not (any(a < 0 for a in amounts) and any(a > 0 for a in amounts)):
        return None

    # Newton
    r = 0.15
    for _ in range(60):
        f = xnpv(r, cf)
        dr = 1e-5
        f2 = xnpv(r + dr, cf)
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

    # Bracket + bisection
    grid = np.concatenate([
        np.linspace(-0.9, 0.0, 30),
        np.linspace(0.0, 1.0, 50),
        np.linspace(1.0, 10.0, 50),
    ])
    prev_r = grid[0]
    prev_f = xnpv(prev_r, cf)
    bracket = None
    for rr in grid[1:]:
        ff = xnpv(rr, cf)
        if np.isfinite(prev_f) and np.isfinite(ff) and (prev_f == 0 or ff == 0 or (prev_f * ff < 0)):
            bracket = (prev_r, rr)
            break
        prev_r, prev_f = rr, ff

    if bracket is None:
        return None

    a, b = bracket
    fa, fb = xnpv(a, cf), xnpv(b, cf)
    if not (np.isfinite(fa) and np.isfinite(fb)):
        return None

    for _ in range(80):
        mid = (a + b) / 2.0
        fm = xnpv(mid, cf)
        if not np.isfinite(fm):
            return None
        if abs(fm) < 1e-8:
            return mid
        if fa * fm < 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return (a + b) / 2.0

# ============================================================
# MARKET TICKER NORMALIZATION
# ============================================================
def normalize_ticker(user_ticker: str, default_market_label: str) -> str:
    t = (user_ticker or "").strip().upper()
    if t == "":
        return t
    if t.startswith("^"):
        return t
    if "." in t:
        return t
    if default_market_label == "NSE (.NS)":
        return f"{t}.NS"
    if default_market_label == "BSE (.BO)":
        return f"{t}.BO"
    return t

# ============================================================
# YFINANCE SAFE FETCH (CACHE + RETRIES)
# ============================================================
@st.cache_data(ttl=86400)
def yf_info_cached(ticker: str):
    for attempt in range(3):
        try:
            time.sleep(0.2 + attempt * 0.2)
            return yf.Ticker(ticker).info or {}
        except Exception:
            time.sleep(0.6 + attempt * 0.8)
    return {}

@st.cache_data(ttl=86400)
def yf_download_cached(symbol: str, start: date, end: date):
    try:
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False, threads=False)
        return df
    except Exception:
        return pd.DataFrame()

def close_on_or_before(prices: pd.Series, d: date):
    if prices is None or prices.empty:
        return None
    ts = pd.Timestamp(d)
    s = prices[prices.index <= ts]
    if s.empty:
        return None
    return float(s.iloc[-1])

# ============================================================
# BUSINESS CLASSIFICATION (BANK/NBFC separated)
# ============================================================
def classify_business(sector: str, industry: str) -> str:
    s = (sector or "").lower()
    i = (industry or "").lower()

    # banks + lenders
    if any(k in i for k in ["bank", "banks"]) or "bank" in s:
        return "BANK"
    if any(k in i for k in ["nbfc", "consumer finance", "credit services", "financial services", "lending", "mortgage"]):
        return "NBFC"
    if any(k in s for k in ["financial"]) and any(k in i for k in ["insurance", "asset management", "capital markets", "brokerage"]):
        return "FINANCIAL"

    # asset-light compounders (approx)
    if any(k in s for k in ["technology", "information technology", "healthcare"]) or any(k in i for k in ["software", "it services", "pharmaceutical", "diagnostics"]):
        return "ASSET_LIGHT"

    # cyclicals (approx)
    if any(k in s for k in ["energy", "basic materials"]) or any(k in i for k in ["steel", "metals", "mining", "oil", "gas", "commodity", "power"]):
        return "CYCLICAL"

    return "GENERAL"

def compute_norm_eps(trailing_eps, forward_eps, override_norm_eps=np.nan):
    o = to_float_safe(override_norm_eps, np.nan)
    if np.isfinite(o) and o > 0:
        return float(o), "override"

    te = to_float_safe(trailing_eps, np.nan)
    fe = to_float_safe(forward_eps, np.nan)

    if np.isfinite(te) and te > 0 and np.isfinite(fe) and fe > 0:
        return float((te + fe) / 2.0), "blend(trailing+forward)"
    if np.isfinite(te) and te > 0:
        return float(te), "trailing"
    if np.isfinite(fe) and fe > 0:
        return float(fe * 0.85), "forward*0.85"
    return 0.0, "missing"

# ============================================================
# INTRINSIC: MULTIPLE MIDPOINT + TIGHT SENSITIVITY BAND
# ============================================================
def intrinsic_multiple(biz: str, roe, bvps, trailing_eps, forward_eps, override_norm_eps, sens_pct: float):
    """
    Returns (min, mid, max, confidence, note, currency_label_unknown_here)
    """
    biz = (biz or "").upper()

    roe_f = to_float_safe(roe, np.nan)
    bvps_f = to_float_safe(bvps, np.nan)

    norm_eps, eps_src = compute_norm_eps(trailing_eps, forward_eps, override_norm_eps)
    sens = float(sens_pct) / 100.0

    # Banks / NBFC -> PB anchor
    if biz in {"BANK", "NBFC"}:
        if not (np.isfinite(bvps_f) and bvps_f > 0 and np.isfinite(roe_f) and roe_f > 0):
            return None, None, None, "Low", "Bank/NBFC: need BVPS and ROE."

        coe = 0.13  # stated assumption for India
        target_pb = max(0.8, min(2.0, roe_f / coe))
        mid = bvps_f * target_pb
        lo = mid * (1.0 - sens)
        hi = mid * (1.0 + sens)

        conf = "High" if roe_f > 0 else "Medium"
        note = f"PB anchor: targetPB=clamp(ROE/CoE,0.8..2.0), CoE={coe*100:.1f}%, band=±{sens_pct}%"
        return lo, mid, hi, conf, note

    # Others -> EPS multiple midpoint with tight band
    if norm_eps <= 0:
        return None, None, None, "Low", "Need EPS (norm/trailing/forward)."

    if biz == "ASSET_LIGHT":
        mult_mid = 22.0
        conf = "High"
        cat = "Asset-light"
    elif biz == "CYCLICAL":
        mult_mid = 10.0
        conf = "Low"
        cat = "Cyclical"
    elif biz == "FINANCIAL":
        mult_mid = 14.0
        conf = "Medium"
        cat = "Financial (non-lender)"
    else:
        mult_mid = 15.0
        conf = "Medium"
        cat = "General"

    mid = norm_eps * mult_mid
    lo = mid * (1.0 - sens)
    hi = mid * (1.0 + sens)

    note = f"{cat}: normEPS({eps_src})×{mult_mid:.1f}, band=±{sens_pct}%"
    return lo, mid, hi, conf, note

def mos_vs_value(price, intrinsic_value):
    p = to_float_safe(price, np.nan)
    v = to_float_safe(intrinsic_value, np.nan)
    if not (np.isfinite(p) and p > 0 and np.isfinite(v) and v > 0):
        return None
    return (v - p) / v * 100.0

# ============================================================
# INTRINSIC: DCF (non-financials only) with net debt adjustment
# ============================================================
def calculate_dcf_intrinsic(
    info: dict,
    shares_outstanding,
    biz_type: str,
    override_fcf=np.nan,
    override_growth_pct=np.nan,
    override_beta=np.nan,
    override_cash=np.nan,
    override_debt=np.nan
):
    """
    Returns (intrinsic_per_share, wacc_pct, growth_pct, confidence, note)
    """
    biz = (biz_type or "").upper()

    # DCF is not meaningful for BANK/NBFC-style balance sheet businesses
    if biz in {"BANK", "NBFC"}:
        return None, None, None, "NA", "DCF disabled for Bank/NBFC."

    # ---- shares ----
    sh = to_float_safe(shares_outstanding, np.nan)
    if not (np.isfinite(sh) and sh > 0):
        return None, None, None, "Low", "Missing shares outstanding."

    # ---- cashflow base ----
    fcf_override = to_float_safe(override_fcf, np.nan)
    if np.isfinite(fcf_override) and fcf_override > 0:
        fcf = float(fcf_override)
        fcf_quality = "override FCF"
    else:
        fcf_raw = info.get("freeCashflow")
        ocf_raw = info.get("operatingCashflow")
        if fcf_raw is None:
            if ocf_raw is None:
                return None, None, None, "Low", "No FCF/OCF data."
            fcf = float(to_float_safe(ocf_raw, np.nan)) * 0.70
            fcf_quality = "OCF×0.70"
        else:
            fcf = float(to_float_safe(fcf_raw, np.nan))
            fcf_quality = "reported FCF"

    if not (np.isfinite(fcf) and fcf > 0):
        return None, None, None, "Low", "FCF not usable (<=0)."

    # ---- beta ----
    b_over = to_float_safe(override_beta, np.nan)
    if np.isfinite(b_over) and b_over > 0:
        beta = float(b_over)
        beta_src = "override"
    else:
        beta = to_float_safe(info.get("beta"), np.nan)
        beta = 1.0 if not (np.isfinite(beta) and beta > 0) else float(beta)
        beta_src = "yfinance/default"

    # ---- RF + market return (simplified) ----
    ccy = (info.get("currency") or "").upper()
    if ccy == "INR":
        risk_free = 0.065
        market_return = 0.12
    else:
        risk_free = 0.04
        market_return = 0.10

    wacc = float(risk_free + beta * (market_return - risk_free))

    # ---- growth ----
    g_over = to_float_safe(override_growth_pct, np.nan)
    if np.isfinite(g_over) and g_over > 0:
        g = float(g_over) / 100.0
        g_src = "override"
    else:
        rg = to_float_safe(info.get("revenueGrowth"), np.nan)
        eg = to_float_safe(info.get("earningsGrowth"), np.nan)
        # yfinance is typically fraction (0.12 = 12%). Reject absurd values.
        rg = rg if (np.isfinite(rg) and -0.5 < rg < 1.5) else np.nan
        eg = eg if (np.isfinite(eg) and -0.5 < eg < 1.5) else np.nan

        if np.isfinite(rg) and rg > 0:
            g = float(rg)
            g_src = "revenueGrowth"
        elif np.isfinite(eg) and eg > 0:
            g = float(eg)
            g_src = "earningsGrowth"
        else:
            g = 0.10
            g_src = "default 10%"

    # clamp growth for sanity
    g = float(min(max(g, 0.05), 0.20))

    terminal_g = 0.03
    if wacc <= terminal_g + 0.03:
        return None, None, None, "Low", "WACC too close to terminal growth."

    years = 5
    pv = 0.0
    last = fcf
    for y in range(1, years + 1):
        last = last * (1 + g)
        pv += last / ((1 + wacc) ** y)

    terminal_cf = last * (1 + terminal_g)
    terminal_value = terminal_cf / (wacc - terminal_g)
    pv_terminal = terminal_value / ((1 + wacc) ** years)
    enterprise_value = pv + pv_terminal

    # net debt adjustment -> equity value
    cash_over = to_float_safe(override_cash, np.nan)
    debt_over = to_float_safe(override_debt, np.nan)

    if np.isfinite(cash_over) and cash_over >= 0:
        cash = float(cash_over)
        cash_src = "override"
    else:
        cash = float(to_float_safe(info.get("totalCash"), 0.0))
        cash_src = "yfinance/0"

    if np.isfinite(debt_over) and debt_over >= 0:
        debt = float(debt_over)
        debt_src = "override"
    else:
        debt = float(to_float_safe(info.get("totalDebt"), 0.0))
        debt_src = "yfinance/0"

    equity_value = enterprise_value + cash - debt
    intrinsic = equity_value / sh

    # confidence
    conf = "Medium"
    if fcf_quality == "reported FCF" and g_src != "default 10%":
        conf = "High"
    if fcf_quality != "reported FCF" or g_src == "default 10%":
        conf = "Low"

    note = (
        f"DCF({years}y): fcf={fcf_quality}, g={g*100:.1f}%({g_src}), "
        f"WACC={wacc*100:.1f}%({ccy or 'ccy?'}), cash={cash_src}, debt={debt_src}, beta={beta_src}"
    )
    return float(intrinsic), wacc * 100, g * 100, conf, note

# ============================================================
# NIFTY BENCHMARK (mirrors buys & sells)
# ============================================================
def compute_nifty_xirr(tx_cashflows, valuation_date: date):
    if not tx_cashflows or len(tx_cashflows) < 2:
        return None, "No portfolio cashflows."

    start = min(d for d, _ in tx_cashflows)
    hist = yf_download_cached("^NSEI", start - timedelta(days=15), valuation_date + timedelta(days=5))
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None, "NIFTY data unavailable (^NSEI)."

    prices = hist["Close"].dropna()
    if prices.empty:
        return None, "NIFTY prices unavailable."

    units = 0.0
    bench_cf = []

    for d, amt in sorted(tx_cashflows, key=lambda x: x[0]):
        amt = float(amt)
        px = close_on_or_before(prices, d)
        if px is None or px <= 0:
            continue

        if amt < 0:
            units += abs(amt) / px
            bench_cf.append((d, amt))
        elif amt > 0:
            sell_units = amt / px
            if sell_units > units:
                sell_units = units
                amt = sell_units * px
            units -= sell_units
            if amt > 0:
                bench_cf.append((d, amt))

    if units <= 0 or len(bench_cf) < 2:
        return None, "Benchmark not computable (insufficient mapped cashflows)."

    final_px = close_on_or_before(prices, valuation_date)
    if final_px is None or final_px <= 0:
        return None, "No NIFTY close for valuation date."

    bench_cf.append((valuation_date, units * final_px))
    r = xirr(bench_cf)
    return r, "Benchmark uses ^NSEI closes; dividends excluded."

# ============================================================
# UI: TEMPLATE
# ============================================================
def make_template():
    return pd.DataFrame([
        {"Ticker": "INFY", "Date": "2019-06-10", "Action": "BUY",  "Quantity": 10, "Price": 700,  "Charges": 0, "CMP": 0},
        {"Ticker": "INFY", "Date": "2020-01-15", "Action": "BUY",  "Quantity": 5,  "Price": 800,  "Charges": 0, "CMP": 0},
        {"Ticker": "INFY", "Date": "2021-05-20", "Action": "SELL", "Quantity": 3,  "Price": 1200, "Charges": 0, "CMP": 0},
        {"Ticker": "HDFCBANK", "Date": "2023-07-01", "Action": "BUY", "Quantity": 2, "Price": 1550, "Charges": 0, "CMP": 0},
    ])

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

with st.expander("🧠 Intrinsic logic (what this app assumes)"):
    st.markdown(
        f"""
- **Multiple-based intrinsic** is computed as a **midpoint** (e.g., EPS×15 for General) with a **tight band** of **±{sensitivity_pct}%**.
  - This avoids the “huge interval” problem and makes **MoS meaningful**.
- **Banks/NBFCs** use a **P/B anchor**: `Intrinsic_mid = BVPS × clamp(ROE/CoE, 0.8..2.0)` with **CoE = 13%** (stated assumption).
- **DCF (optional)** is computed **only for non-financials** and adjusts EV → Equity by **Cash − Debt** when available.
- If Yahoo data is missing/rate-limited, use the **Overrides** table to input EPS/BVPS/ROE/FCF/Shares.
        """
    )

# ============================================================
# MAIN UI
# ============================================================
st.markdown("### Upload your transactions CSV")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if not uploaded:
    st.stop()

raw = pd.read_csv(uploaded)
missing = validate_columns(raw)
if missing:
    st.error(f"Missing columns: {missing}. Please use the template and keep EXACT names.")
    st.stop()

df = clean_transactions(raw)
if df.empty:
    st.error("No valid BUY/SELL rows after cleaning. Check your file.")
    st.stop()

valuation_date = st.date_input("Valuation Date", value=date.today())

st.markdown("### Edit CMP per stock (manual CMP overrides)")
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

# ============================================================
# COMPUTE BUTTON
# ============================================================
if not st.button("🚀 Compute Stock XIRR + Intrinsic + Portfolio vs NIFTY"):
    st.stop()

cmp_map = edited_cmp.set_index("Ticker")["CMP"].to_dict()

# First pass: compute holdings + cashflows, build list of active holdings
tickers = sorted(df["Ticker"].unique().tolist())

holdings = {}
portfolio_tx_cf = []
rows_basic = []

for tkr, g in df.groupby("Ticker"):
    g = g.sort_values("Date").copy()
    qty = 0.0
    invested = 0.0
    realized = 0.0
    cashflows = []
    first_buy = g[g["Action"] == "BUY"]["Date"].min()
    last_txn = g["Date"].max()

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

        cashflows.append((dt, float(amt)))
        portfolio_tx_cf.append((dt, float(amt)))

    cmp_price = float(to_float_safe(cmp_map.get(tkr, 0.0), 0.0))
    current_value = qty * cmp_price if qty > 0 else 0.0
    if qty > 0 and current_value > 0:
        cashflows.append((valuation_date, float(current_value)))

    stock_r = xirr(cashflows)

    holdings[tkr] = {
        "qty": qty,
        "cmp": cmp_price,
        "current_value": current_value,
        "first_buy": first_buy,
        "last_txn": last_txn,
        "invested": invested,
        "realized": realized,
        "stock_xirr": stock_r,
    }

# Active holdings only for intrinsic
active_tickers = [t for t in tickers if holdings.get(t, {}).get("qty", 0) > 0]

# ============================================================
# FUNDAMENTALS PREFILL (Yahoo) + OVERRIDES TABLE
# ============================================================
fund_rows = []
if use_autofetch and active_tickers:
    st.info(f"Fetching fundamentals for {len(active_tickers)} active holdings (cached; throttled).")
    p = st.progress(0)
    for i, tkr in enumerate(active_tickers, start=1):
        yf_tkr = normalize_ticker(tkr, default_market)
        info = yf_info_cached(yf_tkr) if use_autofetch else {}
        time.sleep(float(sleep_between_calls))
        sector = info.get("sector", "") or ""
        industry = info.get("industry", "") or ""
        biz = classify_business(sector, industry)
        currency = info.get("currency", "") or ""

        fund_rows.append({
            "Ticker": tkr,
            "YF Ticker": yf_tkr,
            "Currency": currency,

            "Sector": sector,
            "Industry": industry,
            "Biz Type": biz,

            # inputs from Yahoo (read-only)
            "ROE (fraction) [YF]": to_float_safe(info.get("returnOnEquity"), np.nan),
            "BVPS [YF]": to_float_safe(info.get("bookValue"), np.nan),
            "Trailing EPS [YF]": to_float_safe(info.get("trailingEps"), np.nan),
            "Forward EPS [YF]": to_float_safe(info.get("forwardEps"), np.nan),
            "Shares Outstanding [YF]": to_float_safe(info.get("sharesOutstanding"), np.nan),
            "Free Cash Flow [YF]": to_float_safe(info.get("freeCashflow"), np.nan),
            "Operating Cash Flow [YF]": to_float_safe(info.get("operatingCashflow"), np.nan),
            "Revenue Growth (frac) [YF]": to_float_safe(info.get("revenueGrowth"), np.nan),
            "Earnings Growth (frac) [YF]": to_float_safe(info.get("earningsGrowth"), np.nan),
            "Beta [YF]": to_float_safe(info.get("beta"), np.nan),
            "Total Cash [YF]": to_float_safe(info.get("totalCash"), np.nan),
            "Total Debt [YF]": to_float_safe(info.get("totalDebt"), np.nan),

            # override inputs (editable)
            "Norm EPS [Override]": np.nan,
            "BVPS [Override]": np.nan,
            "ROE (fraction) [Override]": np.nan,
            "Shares Outstanding [Override]": np.nan,

            "FCF [Override]": np.nan,
            "Growth % [Override]": np.nan,
            "Beta [Override]": np.nan,
            "Cash [Override]": np.nan,
            "Debt [Override]": np.nan,
        })
        p.progress(i / len(active_tickers))
else:
    # Build blank rows so user can override even if autofetch is off
    for tkr in active_tickers:
        yf_tkr = normalize_ticker(tkr, default_market)
        fund_rows.append({
            "Ticker": tkr,
            "YF Ticker": yf_tkr,
            "Currency": "",
            "Sector": "",
            "Industry": "",
            "Biz Type": "",

            "ROE (fraction) [YF]": np.nan,
            "BVPS [YF]": np.nan,
            "Trailing EPS [YF]": np.nan,
            "Forward EPS [YF]": np.nan,
            "Shares Outstanding [YF]": np.nan,
            "Free Cash Flow [YF]": np.nan,
            "Operating Cash Flow [YF]": np.nan,
            "Revenue Growth (frac) [YF]": np.nan,
            "Earnings Growth (frac) [YF]": np.nan,
            "Beta [YF]": np.nan,
            "Total Cash [YF]": np.nan,
            "Total Debt [YF]": np.nan,

            "Norm EPS [Override]": np.nan,
            "BVPS [Override]": np.nan,
            "ROE (fraction) [Override]": np.nan,
            "Shares Outstanding [Override]": np.nan,

            "FCF [Override]": np.nan,
            "Growth % [Override]": np.nan,
            "Beta [Override]": np.nan,
            "Cash [Override]": np.nan,
            "Debt [Override]": np.nan,
        })

fund_df = pd.DataFrame(fund_rows)

st.subheader("🧾 Fundamentals & Overrides (only for ACTIVE holdings)")
st.caption(
    "If Yahoo data is missing or rate-limited, fill **Override** columns. "
    "Overrides are used in preference to Yahoo for intrinsic."
)

disabled_cols = [
    "Ticker", "YF Ticker", "Currency", "Sector", "Industry", "Biz Type",
    "ROE (fraction) [YF]", "BVPS [YF]", "Trailing EPS [YF]", "Forward EPS [YF]",
    "Shares Outstanding [YF]", "Free Cash Flow [YF]", "Operating Cash Flow [YF]",
    "Revenue Growth (frac) [YF]", "Earnings Growth (frac) [YF]", "Beta [YF]",
    "Total Cash [YF]", "Total Debt [YF]"
]

edited_fund = st.data_editor(
    fund_df,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    column_config={c: st.column_config.NumberColumn(step=0.01) for c in fund_df.columns if c not in disabled_cols},
    disabled=disabled_cols
)

# ============================================================
# FINAL COMPUTE: intrinsic + outputs
# ============================================================
fund_map = edited_fund.set_index("Ticker").to_dict(orient="index")

result_rows = []
total_current_value = 0.0

for tkr in tickers:
    h = holdings[tkr]
    qty = float(h["qty"])
    cmp_price = float(h["cmp"])
    current_value = float(h["current_value"])
    total_current_value += current_value

    intrinsic_mult_range = "NA"
    intrinsic_mult_mid = None
    mos_mult = None
    conf_mult = "NA"
    note_mult = "NA"

    intrinsic_dcf = None
    mos_dcf = None
    wacc = None
    g_pct = None
    conf_dcf = "NA"
    note_dcf = "NA"

    currency = ""

    if qty > 0:
        fm = fund_map.get(tkr, {})

        sector = fm.get("Sector", "") or ""
        industry = fm.get("Industry", "") or ""
        biz = fm.get("Biz Type", "") or classify_business(sector, industry)
        currency = fm.get("Currency", "") or ""

        # choose inputs with overrides
        roe = fm.get("ROE (fraction) [Override]")
        if not (np.isfinite(to_float_safe(roe, np.nan)) and to_float_safe(roe, np.nan) > 0):
            roe = fm.get("ROE (fraction) [YF]")

        bvps = fm.get("BVPS [Override]")
        if not (np.isfinite(to_float_safe(bvps, np.nan)) and to_float_safe(bvps, np.nan) > 0):
            bvps = fm.get("BVPS [YF]")

        trailing_eps = fm.get("Trailing EPS [YF]")
        forward_eps = fm.get("Forward EPS [YF]")
        norm_eps_override = fm.get("Norm EPS [Override]")

        # MULTIPLE-BASED
        lo, mid, hi, conf_mult, note_mult = intrinsic_multiple(
            biz=biz,
            roe=roe,
            bvps=bvps,
            trailing_eps=trailing_eps,
            forward_eps=forward_eps,
            override_norm_eps=norm_eps_override,
            sens_pct=sensitivity_pct
        )
        if mid is not None and lo is not None and hi is not None:
            intrinsic_mult_mid = float(mid)
            intrinsic_mult_range = f"{lo:,.2f} – {hi:,.2f}"
            mos_mult = mos_vs_value(cmp_price, intrinsic_mult_mid)

        # DCF (optional)
        if use_dcf:
            # shares overrides
            shares = fm.get("Shares Outstanding [Override]")
            if not (np.isfinite(to_float_safe(shares, np.nan)) and to_float_safe(shares, np.nan) > 0):
                shares = fm.get("Shares Outstanding [YF]")

            # DCF overrides
            fcf_over = fm.get("FCF [Override]")
            growth_over = fm.get("Growth % [Override]")
            beta_over = fm.get("Beta [Override]")
            cash_over = fm.get("Cash [Override]")
            debt_over = fm.get("Debt [Override]")

            # build an info dict from YF columns we already have (so DCF can still run without extra .info calls)
            # NOTE: if you disabled autofetch, these may be NaN and DCF will likely become NA unless overrides exist.
            info_like = {
                "currency": currency,
                "freeCashflow": fm.get("Free Cash Flow [YF]"),
                "operatingCashflow": fm.get("Operating Cash Flow [YF]"),
                "revenueGrowth": fm.get("Revenue Growth (frac) [YF]"),
                "earningsGrowth": fm.get("Earnings Growth (frac) [YF]"),
                "beta": fm.get("Beta [YF]"),
                "totalCash": fm.get("Total Cash [YF]"),
                "totalDebt": fm.get("Total Debt [YF]"),
            }

            intrinsic_dcf, wacc, g_pct, conf_dcf, note_dcf = calculate_dcf_intrinsic(
                info=info_like,
                shares_outstanding=shares,
                biz_type=biz,
                override_fcf=fcf_over,
                override_growth_pct=growth_over,
                override_beta=beta_over,
                override_cash=cash_over,
                override_debt=debt_over
            )
            if intrinsic_dcf is not None and np.isfinite(intrinsic_dcf) and intrinsic_dcf > 0:
                mos_dcf = mos_vs_value(cmp_price, intrinsic_dcf)

    result_rows.append({
        "Ticker": tkr,
        "First Buy Date": h["first_buy"],
        "Last Transaction Date": h["last_txn"],
        "Holding Qty (Current)": round(qty, 4),
        "CMP (Editable)": round(cmp_price, 2),
        "Current Value (Qty×CMP)": round(current_value, 2),
        "Total Invested (Buys incl Charges)": round(float(h["invested"]), 2),
        "Total Realized (Sells net Charges)": round(float(h["realized"]), 2),
        "XIRR %": None if h["stock_xirr"] is None else round(float(h["stock_xirr"]) * 100, 2),

        "Currency": currency if qty > 0 else "NA",

        "Intrinsic Range [Multiple]": intrinsic_mult_range if qty > 0 else "NA",
        "Intrinsic Mid [Multiple]": "NA" if (qty <= 0 or intrinsic_mult_mid is None) else round(float(intrinsic_mult_mid), 2),
        "MoS % [Multiple]": None if mos_mult is None else round(float(mos_mult), 2),
        "Confidence [Multiple]": conf_mult if qty > 0 else "NA",
        "Notes [Multiple]": note_mult if qty > 0 else "NA",

        "Intrinsic [DCF]": "NA" if (qty <= 0 or intrinsic_dcf is None) else round(float(intrinsic_dcf), 2),
        "MoS % [DCF]": None if mos_dcf is None else round(float(mos_dcf), 2),
        "WACC % [DCF]": "NA" if wacc is None else round(float(wacc), 2),
        "Growth % [DCF]": "NA" if g_pct is None else round(float(g_pct), 2),
        "Confidence [DCF]": conf_dcf if qty > 0 else "NA",
        "Notes [DCF]": note_dcf if qty > 0 else "NA",
    })

stock_df = pd.DataFrame(result_rows)

# Portfolio XIRR uses single terminal = total current value
portfolio_cf = list(portfolio_tx_cf)
if total_current_value > 0:
    portfolio_cf.append((valuation_date, float(total_current_value)))
portfolio_r = xirr(portfolio_cf)

# NIFTY XIRR uses tx cashflows only (mirrors buys & sells) + its own terminal
nifty_r, nifty_note = compute_nifty_xirr(portfolio_tx_cf, valuation_date)

# ============================================================
# OUTPUT UI
# ============================================================
st.subheader("✅ Portfolio Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Portfolio XIRR", "NA" if portfolio_r is None else f"{portfolio_r*100:.2f}%")
c2.metric("NIFTY 50 XIRR", "NA" if nifty_r is None else f"{nifty_r*100:.2f}%")
c3.metric("Alpha (Portfolio - NIFTY)", "NA" if (portfolio_r is None or nifty_r is None) else f"{(portfolio_r - nifty_r)*100:.2f}%")
c4.metric("Total Current Value", f"{total_current_value:,.2f}")

st.caption(f"NIFTY benchmark note: {nifty_note}")

st.subheader("📌 Stock-wise XIRR + Intrinsic (Active holdings only for intrinsic)")
# show active holdings first by default
stock_df["Is Active Holding"] = stock_df["Holding Qty (Current)"].apply(lambda x: True if to_float_safe(x, 0.0) > 0 else False)
stock_df_display = stock_df.sort_values(["Is Active Holding", "XIRR %"], ascending=[False, False], na_position="last")

st.dataframe(stock_df_display, use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ Download results CSV",
    data=stock_df_display.to_csv(index=False).encode("utf-8"),
    file_name="xirr_intrinsic_results.csv",
    mime="text/csv"
)

st.divider()
st.caption("If you see lots of NA in intrinsic: turn off Auto-fetch (Yahoo) and fill overrides for Norm EPS / BVPS / ROE / Shares / FCF.")

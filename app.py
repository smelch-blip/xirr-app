import time
import traceback
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
st.title("📈 Portfolio XIRR + Intrinsic Valuation + NIFTY Benchmark")

st.warning(
    "⚠️ Corporate actions (splits/bonus) and dividends are NOT auto-captured in this version. "
    "If your quantities/prices are not adjusted manually, XIRR & intrinsic values may be skewed."
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
        return pd.to_datetime(str(x), errors="coerce").date()
    except Exception:
        return None

def to_float_safe(x, default=0.0):
    try:
        if pd.isna(x) or x == "":
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
        df[col] = df[col].apply(to_float_safe)

    # keep only BUY/SELL
    df = df[df["Action"].isin(ACTION_ALLOWED)].copy()

    df = df[df["Ticker"].ne("")].copy()
    df = df[df["Quantity"] > 0].copy()
    df = df[df["Price"] >= 0].copy()

    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df

def build_cmp_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    One CMP per ticker, editable.
    Uses last non-zero CMP seen for that ticker.
    """
    rows = []
    for tkr, g in df.groupby("Ticker"):
        vals = [float(v) for v in g["CMP"].tolist() if v is not None]
        nz = [v for v in vals if v > 0]
        cmp_val = nz[-1] if nz else (vals[-1] if vals else 0.0)
        rows.append({"Ticker": tkr, "CMP": float(cmp_val)})
    return pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)

# ============================================================
# XIRR ENGINE (Newton + Bisection fallback)
# ============================================================

def xnpv(rate, cashflows):
    if rate <= -0.999999:
        return np.inf
    t0 = min(dt for dt, _ in cashflows)
    total = 0.0
    for dt, amt in cashflows:
        amt = float(amt)
        years = (dt - t0).days / 365.25
        total += amt / ((1.0 + rate) ** years)
    return total

def xirr(cashflows):
    """
    Returns rate as decimal (0.15 = 15%), or None if not solvable.
    Uses Newton then bracket+bisection fallback.
    """
    if not cashflows or len(cashflows) < 2:
        return None

    # ensure python floats (avoid pandas ambiguity bugs)
    cf = [(d, float(a)) for d, a in cashflows if d is not None and a is not None]
    if len(cf) < 2:
        return None

    amounts = [a for _, a in cf]
    has_neg = any(a < 0 for a in amounts)
    has_pos = any(a > 0 for a in amounts)
    if not (has_neg and has_pos):
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

def normalize_ticker(user_ticker: str, default_market: str) -> str:
    """
    If user already supplies suffix (.NS/.BO) or special tickers (^...), keep it.
    Otherwise attach based on default market.
    """
    t = (user_ticker or "").strip().upper()
    if t == "":
        return t
    if t.startswith("^"):
        return t
    if "." in t:  # already has suffix
        return t
    if default_market == "NSE (.NS)":
        return f"{t}.NS"
    if default_market == "BSE (.BO)":
        return f"{t}.BO"
    return t  # None

# ============================================================
# YFINANCE SAFE FETCH (RETRIES + CACHE)
# ============================================================

@st.cache_data(ttl=86400)
def yf_info_cached(ticker: str):
    """
    Cached fundamentals. Retries to reduce rate limit failure.
    If it fails, returns {}.
    """
    for attempt in range(3):
        try:
            # small jitter to reduce burst
            time.sleep(0.2 + attempt * 0.2)
            t = yf.Ticker(ticker)
            info = t.info or {}
            return info
        except Exception:
            # backoff
            time.sleep(0.6 + attempt * 0.8)
    return {}

@st.cache_data(ttl=86400)
def yf_download_cached(symbol: str, start: date, end: date):
    """
    Cached price history. Uses yfinance download.
    """
    try:
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False, threads=False)
        return df
    except Exception:
        return pd.DataFrame()

def close_on_or_before(prices: pd.Series, d: date):
    """
    Return last available close on or before date d (handles weekends/holidays).
    """
    if prices is None or prices.empty:
        return None
    ts = pd.Timestamp(d)
    s = prices[prices.index <= ts]
    if s.empty:
        return None
    return float(s.iloc[-1])

# ============================================================
# INTRINSIC VALUATION LOGIC (EPS/BVPS/ROE ANCHORED)
# ============================================================

# Default assumptions (shown in UI)
VAL_RULES = pd.DataFrame([
    {"Category": "Banks",      "Floor": "BVPS × TargetPB × 0.90", "Ceiling": "Floor × 1.25", "Key Input": "BVPS + ROE",      "Confidence": "High"},
    {"Category": "Asset-light","Floor": "Norm EPS × 15",          "Ceiling": "Norm EPS × 28", "Key Input": "Norm EPS",         "Confidence": "High"},
    {"Category": "Cyclicals",  "Floor": "Norm EPS × 8",           "Ceiling": "Norm EPS × 14", "Key Input": "Norm EPS",         "Confidence": "Low"},
    {"Category": "General",    "Floor": "Norm EPS × 12",          "Ceiling": "Norm EPS × 18", "Key Input": "Norm EPS",         "Confidence": "Medium"},
])

def classify_business(sector: str, industry: str) -> str:
    s = (sector or "").lower()
    i = (industry or "").lower()
    # bank detection should use industry too
    if "bank" in i or "banks" in i or "bank" in s:
        return "BANK"
    if any(k in s for k in ["technology", "information technology"]) or any(k in i for k in ["software", "it services"]):
        return "ASSET_LIGHT"
    if any(k in s for k in ["energy", "basic materials"]) or any(k in i for k in ["steel", "metals", "mining", "oil", "gas", "commodity", "power"]):
        return "CYCLICAL"
    if any(k in i for k in ["pharmaceutical", "healthcare", "diagnostics"]):
        return "ASSET_LIGHT"
    return "GENERAL"

def compute_norm_eps(trailing_eps, forward_eps):
    trailing_eps = to_float_safe(trailing_eps, 0.0)
    forward_eps  = to_float_safe(forward_eps, 0.0)

    if trailing_eps > 0 and forward_eps > 0:
        return (trailing_eps + forward_eps) / 2.0
    if trailing_eps > 0:
        return trailing_eps
    if forward_eps > 0:
        # forward only -> be conservative
        return forward_eps * 0.85
    return 0.0

def intrinsic_from_inputs(biz: str, roe: float, bvps: float, norm_eps: float):
    """
    Returns (floor, ceiling, confidence, used_inputs_dict, note)
    """
    roe = to_float_safe(roe, 0.0)  # yfinance ROE often as fraction e.g., 0.18
    bvps = to_float_safe(bvps, 0.0)
    norm_eps = to_float_safe(norm_eps, 0.0)

    used = {"biz": biz, "roe": roe, "bvps": bvps, "norm_eps": norm_eps}
    note = ""

    if biz == "BANK":
        if bvps > 0 and roe > 0:
            coe = 0.13
            target_pb = max(0.8, min(2.0, roe / coe))  # ROE & CoE in same units (fraction)
            floor = bvps * target_pb * 0.90
            ceiling = floor * 1.25
            return floor, ceiling, "High", used, "Bank PB anchor (ROE/CoE)."
        return None, None, "Low", used, "Missing BVPS/ROE."

    # Non-banks need EPS
    if norm_eps <= 0:
        return None, None, "Low", used, "Missing EPS."

    if biz == "ASSET_LIGHT":
        floor, ceiling, conf = norm_eps * 15.0, norm_eps * 28.0, "High"
        return floor, ceiling, conf, used, "EPS multiple band for asset-light compounders."
    if biz == "CYCLICAL":
        floor, ceiling, conf = norm_eps * 8.0, norm_eps * 14.0, "Low"
        return floor, ceiling, conf, used, "Mid-cycle EPS multiple band (cyclical)."
    # GENERAL
    floor, ceiling, conf = norm_eps * 12.0, norm_eps * 18.0, "Medium"
    return floor, ceiling, conf, used, "General EPS multiple band."

def margin_of_safety_vs_mid(cmp_price, floor, ceiling):
    """
    MoS vs intrinsic midpoint: (mid - price)/mid
    """
    cmp_price = to_float_safe(cmp_price, np.nan)
    if not np.isfinite(cmp_price) or cmp_price <= 0:
        return None
    if floor is None or ceiling is None:
        return None
    mid = (floor + ceiling) / 2.0
    if mid <= 0:
        return None
    return (mid - cmp_price) / mid * 100.0

# ============================================================
# NIFTY BENCHMARK (BUY + SELL MIRROR)
# ============================================================

def compute_nifty_xirr(tx_cashflows, valuation_date: date):
    """
    tx_cashflows: list[(date, amt)] - ONLY transaction cashflows (no terminal).
    Benchmark mirrors buys (amt<0) and sells (amt>0) into units of NIFTY.
    Then adds terminal value: units * nifty_close(valuation_date).
    """
    if not tx_cashflows or len(tx_cashflows) < 2:
        return None, "No portfolio cashflows"

    start = min(d for d, _ in tx_cashflows)
    # pull a little extra range
    hist = yf_download_cached("^NSEI", start - timedelta(days=15), valuation_date + timedelta(days=5))
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None, "NIFTY data unavailable (^NSEI)."

    prices = hist["Close"].dropna()
    if prices.empty:
        return None, "NIFTY prices unavailable."

    units = 0.0
    bench_cf = []

    # Important: mirror all cashflows
    for d, amt in sorted(tx_cashflows, key=lambda x: x[0]):
        amt = float(amt)

        px = close_on_or_before(prices, d)
        if px is None or px <= 0:
            # skip if we truly can't map date -> close
            continue

        if amt < 0:
            # buy units
            units += abs(amt) / px
            bench_cf.append((d, amt))
        elif amt > 0:
            # sell units to match cashflow withdrawal
            sell_units = amt / px
            # cap (can't sell more than you have in the simulated benchmark)
            if sell_units > units:
                sell_units = units
                # adjust cashflow down to what could be sold
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
# MAIN COMPUTE: STOCK XIRR + PORTFOLIO XIRR + INTRINSIC
# ============================================================

def compute_all(df: pd.DataFrame, cmp_override_df: pd.DataFrame, valuation_date: date, default_market: str, use_autofetch: bool):
    # CMP overrides
    cmp_map = cmp_override_df.set_index("Ticker")["CMP"].to_dict()

    rows = []

    # Transaction cashflows for portfolio (NO terminal here)
    portfolio_tx_cf = []

    # Track current total value (for single terminal cashflow)
    total_current_value = 0.0

    # Collect active holdings tickers (for fundamentals, if enabled)
    # We'll compute per-stock first
    fundamentals_inputs = []

    # Optional: collect per ticker fundamentals now (but do NOT fail if rate-limited)
    # We will show an editable table after compute as well
    # In this function we only compute intrinsic from available info.

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

        cmp_price = float(cmp_map.get(tkr, 0.0))
        current_value = qty * cmp_price if qty > 0 else 0.0

        # stock-level terminal cashflow (only if active holding)
        if qty > 0 and current_value > 0:
            cashflows.append((valuation_date, float(current_value)))
            total_current_value += float(current_value)

        stock_r = xirr(cashflows)

        # -------- Intrinsic (use auto-fetch if enabled, else NA until user overrides) --------
        floor = ceiling = None
        conf = "Low"
        used_inputs = {}
        val_note = ""

        if qty > 0:  # only for active holdings
            if use_autofetch:
                yf_ticker = normalize_ticker(tkr, default_market)
                info = yf_info_cached(yf_ticker)

                sector = info.get("sector", "") or ""
                industry = info.get("industry", "") or ""
                roe = info.get("returnOnEquity", None)
                bvps = info.get("bookValue", None)
                trailing_eps = info.get("trailingEps", None)
                forward_eps = info.get("forwardEps", None)

                biz = classify_business(sector, industry)
                norm_eps = compute_norm_eps(trailing_eps, forward_eps)

                floor, ceiling, conf, used_inputs, val_note = intrinsic_from_inputs(
                    biz=biz,
                    roe=roe,
                    bvps=bvps,
                    norm_eps=norm_eps
                )

                # Keep a copy for potential override editing UI
                fundamentals_inputs.append({
                    "Ticker": tkr,
                    "YF Ticker": yf_ticker,
                    "Sector": sector,
                    "Industry": industry,
                    "Biz Type": biz,
                    "ROE (fraction)": None if roe is None else float(roe),
                    "BVPS": None if bvps is None else float(bvps),
                    "Trailing EPS": None if trailing_eps is None else float(trailing_eps),
                    "Forward EPS": None if forward_eps is None else float(forward_eps),
                    "Norm EPS (used)": float(norm_eps) if norm_eps else 0.0,
                })
            else:
                # still create row for override UI
                fundamentals_inputs.append({
                    "Ticker": tkr,
                    "YF Ticker": normalize_ticker(tkr, default_market),
                    "Sector": "",
                    "Industry": "",
                    "Biz Type": "",
                    "ROE (fraction)": None,
                    "BVPS": None,
                    "Trailing EPS": None,
                    "Forward EPS": None,
                    "Norm EPS (used)": None,
                })
                val_note = "Auto-fetch disabled. Use overrides to compute intrinsic."

        mos = margin_of_safety_vs_mid(cmp_price, floor, ceiling)
        mid = None if (floor is None or ceiling is None) else (floor + ceiling) / 2.0

        intrinsic_range = "NA"
        if floor is not None and ceiling is not None:
            intrinsic_range = f"{floor:,.2f} – {ceiling:,.2f}"

        rows.append({
            "Ticker": tkr,
            "First Buy Date": first_buy,
            "Last Transaction Date": last_txn,
            "Holding Qty": round(qty, 4),
            "CMP": round(cmp_price, 2),
            "Current Value": round(current_value, 2),
            "Total Invested": round(invested, 2),
            "Total Realized": round(realized, 2),
            "XIRR %": None if stock_r is None else round(stock_r * 100, 2),

            # Intrinsic columns (only meaningful for active holdings)
            "Intrinsic Range (₹)": intrinsic_range if qty > 0 else "NA",
            "Intrinsic Mid (₹)": "NA" if qty <= 0 or mid is None else round(mid, 2),
            "MoS % (vs Mid)": None if mos is None else round(mos, 2),
            "Valuation Confidence": conf if qty > 0 else "NA",
            "Valuation Note": val_note if qty > 0 else "NA",
        })

    stock_df = pd.DataFrame(rows)

    # ---- Portfolio terminal cashflow (single) ----
    portfolio_cf = list(portfolio_tx_cf)
    if total_current_value > 0:
        portfolio_cf.append((valuation_date, float(total_current_value)))

    portfolio_r = xirr(portfolio_cf)

    # ---- NIFTY benchmark (uses tx cashflows only, mirrors buys+sells) ----
    nifty_r, nifty_note = compute_nifty_xirr(portfolio_tx_cf, valuation_date)

    return stock_df, portfolio_r, total_current_value, nifty_r, nifty_note, pd.DataFrame(fundamentals_inputs)

# ============================================================
# UI: TEMPLATE
# ============================================================

def make_template():
    return pd.DataFrame([
        {"Ticker": "INFY", "Date": "2019-06-10", "Action": "BUY",  "Quantity": 10, "Price": 700,  "Charges": 0, "CMP": 0},
        {"Ticker": "INFY", "Date": "2020-01-15", "Action": "BUY",  "Quantity": 5,  "Price": 800,  "Charges": 0, "CMP": 0},
        {"Ticker": "INFY", "Date": "2021-05-20", "Action": "SELL", "Quantity": 3,  "Price": 1200, "Charges": 0, "CMP": 0},
        {"Ticker": "HDFCBANK",  "Date": "2023-07-01", "Action": "BUY",  "Quantity": 2,  "Price": 1500, "Charges": 0, "CMP": 0},
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

with st.expander("📌 Intrinsic valuation assumptions (defaults)"):
    st.dataframe(VAL_RULES, use_container_width=True, hide_index=True)
    st.caption(
        "Intrinsic is computed from EPS/BVPS/ROE (independent of CMP). "
        "MoS% is computed vs Intrinsic Mid (midpoint of the intrinsic band)."
    )

# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("Settings")
default_market = st.sidebar.selectbox(
    "Default market for plain tickers",
    ["NSE (.NS)", "BSE (.BO)", "None (as-is)"],
    index=0
)
use_autofetch = st.sidebar.toggle(
    "Auto-fetch fundamentals (yfinance) for intrinsic",
    value=True,
    help="If you hit rate limits or data is missing, intrinsic may show NA. You can still fill overrides manually."
)

# ============================================================
# UI: UPLOAD + COMPUTE
# ============================================================

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

    st.markdown("### Edit CMP per stock (manual input)")
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

    if st.button("🚀 Compute XIRR & Intrinsic"):
        try:
            stock_df, portfolio_r, total_cv, nifty_r, nifty_note, fundamentals_df = compute_all(
                df=df,
                cmp_override_df=edited_cmp,
                valuation_date=valuation_date,
                default_market=default_market,
                use_autofetch=use_autofetch
            )

            # Stock table
            st.subheader("📌 Stock-wise XIRR + Intrinsic (single table)")
            # Sort: active holdings first, then by XIRR desc
            stock_df["_active"] = stock_df["Holding Qty"].apply(lambda x: 1 if float(x) > 0 else 0)
            stock_df = stock_df.sort_values(["_active", "XIRR %"], ascending=[False, False], na_position="last").drop(columns=["_active"])
            st.dataframe(stock_df, use_container_width=True, hide_index=True)

            # Download
            st.download_button(
                "⬇️ Download results CSV",
                data=stock_df.to_csv(index=False).encode("utf-8"),
                file_name="stock_xirr_intrinsic_results.csv",
                mime="text/csv"
            )

            st.divider()

            # Portfolio vs NIFTY
            st.subheader("📊 Portfolio vs NIFTY 50")
            c1, c2, c3 = st.columns(3)
            c1.metric("Portfolio XIRR", "NA" if portfolio_r is None else f"{portfolio_r*100:.2f}%")
            c2.metric("NIFTY XIRR", "NA" if nifty_r is None else f"{nifty_r*100:.2f}%")
            alpha = None if (portfolio_r is None or nifty_r is None) else (portfolio_r - nifty_r) * 100.0
            c3.metric("Alpha", "NA" if alpha is None else f"{alpha:.2f}%")
            st.caption(nifty_note)

            st.caption(
                f"Portfolio terminal value used once (valuation date): ₹{total_cv:,.0f}. "
                "Benchmark mirrors BUY and SELL cashflows into NIFTY units; dividends excluded."
            )

            st.divider()

            # Optional: show fetched fundamentals for transparency + manual override guidance
            with st.expander("🔎 Fundamentals used (for intrinsic) — transparency/debug"):
                st.dataframe(fundamentals_df, use_container_width=True, hide_index=True)
                st.caption(
                    "If many values are missing due to Yahoo/yfinance limitations, intrinsic will show NA/Low. "
                    "Best practice: manually override EPS/BVPS/ROE in a future enhancement."
                )

        except Exception:
            st.error("A critical error occurred during computation.")
            st.code(traceback.format_exc())

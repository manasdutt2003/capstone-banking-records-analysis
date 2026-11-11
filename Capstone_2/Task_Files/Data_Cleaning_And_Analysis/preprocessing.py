"""Shared preprocessing helpers for Bank Records 1.xlsx.

Used by solutions.ipynb (Section A) and visualisation.ipynb (Section B).
Same logic lives inline in the notebooks; this module is the importable
canonical copy exercised by test_preprocessing.py.

Covers the Capstone-1 review fixes:
  * B1/B2/B5/B6/B7/B8 - currency labels profiled, ISO 4217 captured per row
    BEFORE numeric conversion, documented rates dict + single target currency.
  * D2/D6             - aml_flag / suspicious_txn_flag unified to 0/1.
  * C5/C6/C8          - parsed dates validated against year/month columns,
                        day/month swap candidates and future dates flagged.
  * region            - one explicit mapping reused everywhere.
"""
import datetime
import re

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Currency handling (B1/B5/B7)
# --------------------------------------------------------------------------

# Known labels -> ISO 4217. Includes the two mojibake sequences that appear
# in the spreadsheet (UTF-8 bytes of the euro / rupee signs decoded as cp1252:
# U+00E2 U+201A U+00AC = 'â‚¬' (euro) and U+00E2 U+201A U+00B9 = 'â‚¹' (rupee)).
SYMBOL_TO_ISO = {
    "\u20ac": "EUR", "\u00e2\u201a\u00ac": "EUR",
    "\u20b9": "INR", "\u00e2\u201a\u00b9": "INR",
    "INR": "INR", "Rs": "INR", "RS": "INR",
    "$": "USD", "US$": "USD", "USD": "USD",
}
# Plain numbers carry no currency label. They are assumed to be in the target
# currency so downstream maths works, but every such row is flagged with this
# code so the assumption stays visible (B8).
UNLABELLED = "UNLABELLED"

# Single target currency + documented conversion rates (B7).
TARGET_CCY = "INR"
RATES = {"INR": 1.0, "USD": 83.5, "EUR": 90.4}
RATES_SOURCE = "RBI reference rates as of 2025-01-01 (static assumption for this exercise)"
RATES_DATE = "2025-01-01"


def extract_currency(val):
    """Return the ISO 4217 code for a raw money cell, before any numeric conversion.

    UNLABELLED is returned for plain numbers; np.nan for missing cells.
    """
    if val is None:
        return np.nan
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return UNLABELLED if not pd.isna(val) else np.nan
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if not s:
        return np.nan
    for sym, iso in SYMBOL_TO_ISO.items():
        if sym in s:
            return iso
    if re.match(r"^[\-\+]?[0-9,.]", s):
        return UNLABELLED
    return "OTHER"


def convert_to_target(amount, iso):
    """Convert *amount* from *iso* currency into TARGET_CCY using RATES.

    UNLABELLED amounts are treated as already being in TARGET_CCY (the
    *_currency column keeps the assumption visible). Unknown codes -> NaN.
    """
    if amount is None or pd.isna(amount) or iso is None or pd.isna(iso):
        return np.nan
    if iso == UNLABELLED:
        iso = TARGET_CCY
    rate = RATES.get(str(iso).upper())
    if rate is None:
        return np.nan
    return float(amount) * rate


def to_num(val):
    """Turn values like '€1,234.5', 'INR 200', '$10k', '₹500' into numbers."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return val
    s = str(val).strip()
    mult = 1
    if s.lower().endswith('k'):
        mult = 1000
        s = s[:-1]
    s = re.sub(r'[^0-9.\-]', '', s)   # drop currency symbols & text
    try:
        return float(s) * mult
    except ValueError:
        return np.nan


def to_date(v):
    if pd.isna(v):
        return pd.NaT
    if isinstance(v, (datetime.datetime, datetime.date, pd.Timestamp, np.datetime64)):
        return pd.to_datetime(v)
    if isinstance(v, (int, float)):
        s = str(int(v))
    else:
        s = str(v).strip()
    # dates like 'February4-February-2025'
    m = re.match(r'([A-Za-z]+)(\d{1,2})-[A-Za-z]+-(\d{4})', s)
    if m:
        return pd.to_datetime(m.group(1) + m.group(2) + m.group(3),
                              format='%B%d%Y', errors='coerce')
    # compact numbers like '28122026' (DDMMYYYY) or 6-digit DMMYY
    if s.isdigit():
        if len(s) == 8:                       # DDMMYYYY (last 4 = year)
            y, rest = s[-4:], s[:-4]
        elif len(s) == 6:                     # DMMYY -> assume 20YY
            rest, y = s[:2], '20' + s[-2:]
            return pd.to_datetime(rest[0] + '/' + rest[1] + '/' + y,
                                  dayfirst=True, errors='coerce')
        else:
            y, rest = s[-4:], s[:-4]
        try:
            if len(rest) == 4:                 # DDMM
                d, mth = rest[:2], rest[2:]
            elif len(rest) == 3:               # DDM
                d, mth = rest[:2], rest[2]
            else:                              # single-digit day/month
                d, mth = rest[0], rest[-1]
            return pd.to_datetime(f"{d}/{mth}/{y}", dayfirst=True, errors='coerce')
        except Exception:
            return pd.NaT
    # ISO-like 'YYYY-MM-DD [HH:MM:SS]' must NOT use dayfirst
    if re.match(r'^\d{4}-\d{1,2}-\d{1,2}', s):
        return pd.to_datetime(s, errors='coerce')
    return pd.to_datetime(s, dayfirst=True, errors='coerce')


# --------------------------------------------------------------------------
# Flag unification (D2/D6)
# --------------------------------------------------------------------------

_TRUE_STR = {"yes", "y", "true", "1", "100%"}
_FALSE_STR = {"no", "n", "false", "0", "0%"}


def unify_flag(v):
    """Map mixed encodings (1/0, 'Yes'/'No', '100%'/'0%') to 1/0, else pd.NA."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return pd.NA
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _TRUE_STR:
            return 1
        if s in _FALSE_STR:
            return 0
        return pd.NA
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return pd.NA
    if fv == 1.0:
        return 1
    if fv == 0.0:
        return 0
    return pd.NA


# --------------------------------------------------------------------------
# Region standardisation (one mapping, reused by every chart)
# --------------------------------------------------------------------------

# Documented assumptions for the non-obvious aliases: W1/W2/W3 are west-zone
# codes; SNorth reads as 'South'; ENorth reads as 'East-North' -> North-East;
# Upper/Lower/Middle Region map to North/South/Central.
REGION_MAP = {
    "North": "North", "South": "South", "East": "East", "West": "West",
    "North-East": "North-East", "north east": "North-East", "ENorth": "North-East",
    "South-East": "South-East", "south east": "South-East",
    "South-West": "South-West", "US - North West": "North-West",
    "W1": "West", "W2": "West", "W3": "West",
    "SNorth": "South", "Upper Region": "North", "Lower Region": "South",
    "Middle Region": "Central",
}


def standardise_region(v):
    """Apply the single REGION_MAP; unknown labels pass through stripped."""
    if v is None or pd.isna(v):
        return np.nan
    s = str(v).strip()
    if s in REGION_MAP:
        return REGION_MAP[s]
    titled = s.title()
    return REGION_MAP.get(titled, s)


# --------------------------------------------------------------------------
# Date validation helpers (C5/C6/C8)
# --------------------------------------------------------------------------

def month_mismatch(parsed, month_col):
    """Count rows where parsed month != provided month column (hidden swaps)."""
    p = pd.Series(parsed)
    m = pd.Series(month_col)
    return int((p.notna() & m.notna() & (p.dt.month != m)).sum())


def future_flags(parsed, as_of=None):
    """Boolean Series marking dates later than *as_of* (default: today)."""
    as_of = pd.Timestamp(as_of or datetime.date.today())
    return pd.Series(parsed) > as_of


def swap_candidates(raw_series):
    """Count ambiguous numeric dates that parse differently day-first vs month-first."""
    import warnings
    rs = pd.Series(raw_series)
    dfirst = rs.apply(to_date)
    with warnings.catch_warnings():
        # We intentionally parse month-first here to compare against day-first.
        warnings.simplefilter('ignore', UserWarning)
        mfirst = rs.apply(
            lambda v: pd.to_datetime(v, errors='coerce') if isinstance(v, str) else pd.NaT)
    ambiguous = rs.astype(str).str.match(r'^\d{1,2}/\d{1,2}/\d{2,4}$')
    return int((ambiguous & dfirst.notna() & mfirst.notna() & (dfirst != mfirst)).sum())

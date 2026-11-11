"""Tests for preprocessing helpers (Section A).

Covers to_num / to_date plus the review-fix helpers: extract_currency,
convert_to_target, unify_flag, standardise_region and the date-validation
helpers (month_mismatch / future_flags / swap_candidates).

Run: pytest test_preprocessing.py -v
"""
import datetime

import numpy as np
import pandas as pd

from preprocessing import (
    RATES,
    TARGET_CCY,
    UNLABELLED,
    convert_to_target,
    extract_currency,
    future_flags,
    month_mismatch,
    standardise_region,
    swap_candidates,
    to_date,
    to_num,
    unify_flag,
)


def test_to_num_currency_symbols():
    assert to_num('€1,234.5') == 1234.5
    assert to_num('INR 200') == 200.0
    assert to_num('$10k') == 10000.0
    assert to_num('₹500') == 500.0
    assert to_num('USD 68.78') == 68.78
    assert to_num('€-1354.76') == -1354.76   # minus sign preserved (B2)


def test_to_num_passthrough_and_missing():
    assert to_num(42) == 42
    assert to_num(3.5) == 3.5
    assert pd.isna(to_num(None))
    assert pd.isna(to_num(np.nan))
    assert pd.isna(to_num('not-a-number'))
    assert pd.isna(to_num(''))


def test_to_num_dataframe_column():
    s = pd.Series(['€1,234.5', 'INR 200', '$10k', None])
    out = s.apply(to_num)
    assert list(out[:3]) == [1234.5, 200.0, 10000.0]
    assert pd.isna(out[3])


def test_to_date_named_compact_and_iso():
    assert to_date('February4-February-2025').date() == datetime.date(2025, 2, 4)
    assert to_date('15/07/2022').date() == datetime.date(2022, 7, 15)
    assert to_date('2024-10-01 05:17:00').date() == datetime.date(2024, 10, 1)
    assert to_date('28122026').date() == datetime.date(2026, 12, 28)


def test_to_date_passthrough_and_missing():
    assert to_date(datetime.date(2024, 1, 2)).date() == datetime.date(2024, 1, 2)
    assert to_date(pd.Timestamp('2023-05-06')).date() == datetime.date(2023, 5, 6)
    assert pd.isna(to_date(None))
    assert pd.isna(to_date('not-a-date'))


def test_to_date_dataframe_column():
    s = pd.Series(['February4-February-2025', '15/07/2022', None])
    out = s.apply(to_date)
    assert out[0].date() == datetime.date(2025, 2, 4)
    assert out[1].date() == datetime.date(2022, 7, 15)
    assert pd.isna(out[2])


# --------------------------------------------------------------------------
# Currency ISO capture + conversion (B1/B5/B7)
# --------------------------------------------------------------------------

def test_extract_currency_labels():
    assert extract_currency('€1,234.5') == 'EUR'
    assert extract_currency('₹872.32') == 'INR'
    assert extract_currency('INR 1799.68') == 'INR'
    assert extract_currency('USD 68.78') == 'USD'
    assert extract_currency('$10k') == 'USD'


def test_extract_currency_mojibake_euro_and_rupee():
    # The spreadsheet stores UTF-8 euro/rupee bytes decoded as cp1252.
    assert extract_currency('\u00e2\u201a\u00ac8561.29') == 'EUR'
    assert extract_currency('\u00e2\u201a\u00b92811.49') == 'INR'


def test_extract_currency_unlabelled_and_missing():
    assert extract_currency('8216.04') == UNLABELLED
    assert extract_currency(10000) == UNLABELLED
    assert pd.isna(extract_currency(None))
    assert pd.isna(extract_currency(np.nan))


def test_convert_to_target_uses_rates():
    assert convert_to_target(100.0, 'USD') == 100.0 * RATES['USD']
    assert convert_to_target(100.0, 'EUR') == 100.0 * RATES['EUR']
    assert convert_to_target(100.0, TARGET_CCY) == 100.0
    # Unlabelled numbers are assumed to already be in the target currency.
    assert convert_to_target(500.0, UNLABELLED) == 500.0
    assert pd.isna(convert_to_target(100.0, 'OTHER'))
    assert pd.isna(convert_to_target(np.nan, 'USD'))


# --------------------------------------------------------------------------
# Flag unification (D2/D6)
# --------------------------------------------------------------------------

def test_unify_flag_mixed_encodings():
    assert unify_flag(1) == 1
    assert unify_flag(0) == 0
    assert unify_flag('Yes') == 1
    assert unify_flag('No') == 0
    assert unify_flag('100%') == 1
    assert unify_flag('0%') == 0
    assert pd.isna(unify_flag(None))
    assert pd.isna(unify_flag(np.nan))
    assert pd.isna(unify_flag('maybe'))


# --------------------------------------------------------------------------
# Region standardisation (single mapping, reused everywhere)
# --------------------------------------------------------------------------

def test_standardise_region_aliases():
    assert standardise_region('south east') == 'South-East'
    assert standardise_region('north east') == 'North-East'
    assert standardise_region('ENorth') == 'North-East'
    assert standardise_region('SNorth') == 'South'
    assert standardise_region('W2') == 'West'
    assert standardise_region('US - North West') == 'North-West'
    assert standardise_region('Middle Region') == 'Central'
    assert standardise_region('East') == 'East'
    assert pd.isna(standardise_region(None))


# --------------------------------------------------------------------------
# Date validation (C5/C6/C8)
# --------------------------------------------------------------------------

def test_month_mismatch_detects_swapped_month():
    parsed = pd.Series([pd.Timestamp('2022-07-15'), pd.Timestamp('2024-02-18')])
    month_col = pd.Series([7, 3])   # second row's month disagrees
    assert month_mismatch(parsed, month_col) == 1
    assert month_mismatch(parsed, pd.Series([7, 2])) == 0


def test_future_flags():
    parsed = pd.Series([pd.Timestamp('2020-01-01'), pd.Timestamp('2999-12-31')])
    flags = future_flags(parsed, as_of='2026-01-01')
    assert list(flags) == [False, True]


def test_swap_candidates_counts_ambiguous_dates():
    # '15/07/2022' is unambiguous (day 15); '03/04/2021' parses differently
    # day-first vs month-first and therefore counts as a swap candidate.
    assert swap_candidates(pd.Series(['15/07/2022', '03/04/2021', None])) == 1

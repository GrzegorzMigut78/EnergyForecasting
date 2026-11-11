import datetime as dt

import pandas as pd
import pytest

from prognozowanie.config import CalendarConfig, HolidayFeature
from prognozowanie.data.calendar import build_calendar
from tests.conftest import needs_data


def _day(frame, d):
    return frame.loc[pd.Timestamp(d)]


@needs_data
def test_range_and_corrections(dataset):
    f = dataset.frame
    assert f.index.min() == pd.Timestamp("2001-11-01 00:00")
    assert f.index.max() == pd.Timestamp("2003-06-30 23:00")
    assert len(f) == 607 * 24
    fixed = f.loc["2002-02-01 00:00"]
    assert fixed["load_raw"] < 490 and fixed["load"] > 590
    assert {e["type"] for e in dataset.log} == {"exclude", "fix"}


@needs_data
def test_calendar_matches_review(dataset):
    f = dataset.frame
    assert _day(f, "2003-06-19")["holiday"] == 1  # Boże Ciało
    assert _day(f, "2003-05-19")["holiday"] == 0  # zwykły poniedziałek (błąd w oryginale)
    assert _day(f, "2002-12-24")["holiday"] == 1  # Wigilia
    easter = _day(f, "2002-03-31")
    assert easter["holiday"] == 0 and easter["holiday_statutory"] == 1
    assert _day(f, "2003-06-20")["bridge"] == 1


def test_calendar_rules_without_data():
    cfg = CalendarConfig(
        country="PL",
        timezone="Europe/Warsaw",
        holiday_feature=HolidayFeature(exclude_weekdays=[6], extra_month_days=["12-24"]),
    )
    idx = pd.date_range("2002-03-29", "2002-04-02", freq="D")
    cal = build_calendar(idx, cfg)
    assert cal.loc["2002-03-31", "dst_day"] == 1
    assert cal.loc["2002-04-01", "holiday"] == 1  # poniedziałek wielkanocny
    assert cal.loc["2002-03-30", "holiday_eve"] == 1


def test_gap_in_series_is_rejected(tmp_path):
    from prognozowanie.data.dataset import _check_hourly

    idx = pd.date_range("2020-01-01", periods=48, freq="h").delete(10)
    df = pd.DataFrame({"load": 1.0, "temp": 1.0}, index=idx)
    with pytest.raises(ValueError, match="ciągły"):
        _check_hourly(df)


def test_holiday_library_has_historic_rules():
    """Trzech Króli jest dniem wolnym dopiero od 2011 r."""
    cfg = CalendarConfig(country="PL")
    cal = build_calendar(pd.date_range("2002-01-06", periods=1), cfg)
    assert cal.loc["2002-01-06", "holiday_statutory"] == 0
    cal = build_calendar(pd.date_range("2012-01-06", periods=1), cfg)
    assert cal.loc["2012-01-06", "holiday_statutory"] == 1
    assert dt.date(2002, 1, 6).weekday() == 6

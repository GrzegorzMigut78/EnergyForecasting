"""Kalendarz: święta, zmienna 'święto' w definicji z artykułu, dni pomostowe, zmiana czasu."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import holidays
import pandas as pd

from prognozowanie.config import CalendarConfig


def statutory_holidays(cfg: CalendarConfig, years: range) -> set[dt.date]:
    """Święta ustawowe z biblioteki holidays (stan prawny z danego roku)."""
    cal = holidays.country_holidays(cfg.country, subdiv=cfg.subdivision, years=list(years))
    return set(cal.keys())


def extra_days(cfg: CalendarConfig, years: range) -> set[dt.date]:
    out = set()
    for md in cfg.holiday_feature.extra_month_days:
        month, day = (int(x) for x in md.split("-"))
        out |= {dt.date(y, month, day) for y in years}
    return out


def build_calendar(dates: pd.DatetimeIndex, cfg: CalendarConfig) -> pd.DataFrame:
    """Zmienne kalendarzowe na poziomie doby.

    Kolumny:
      holiday            definicja z artykułu: święto ustawowe poza wykluczonymi dniami
                         tygodnia (niedziela) plus dni dodatkowe (Wigilia)
      holiday_statutory  wszystkie święta ustawowe, także niedzielne
      holiday_eve        dzień przed świętem ustawowym (sam niebędący świętem)
      holiday_after      dzień po święcie ustawowym (sam niebędący świętem)
      bridge             dzień roboczy między dwoma dniami wolnymi, przy święcie
      dst_day            doba zmiany czasu w strefie cfg.timezone
    """
    days = pd.DatetimeIndex(sorted(set(dates.normalize())))
    years = range(days.min().year - 1, days.max().year + 2)
    stat = statutory_holidays(cfg, years)
    extra = extra_days(cfg, years)
    excl = set(cfg.holiday_feature.exclude_weekdays)

    def is_holiday(d: dt.date) -> bool:
        return (d in stat and d.weekday() not in excl) or d in extra

    def non_working(d: dt.date) -> bool:
        return d.weekday() >= 5 or d in stat or d in extra

    rows = []
    for ts in days:
        d = ts.date()
        prev, nxt = d - dt.timedelta(days=1), d + dt.timedelta(days=1)
        bridge = (
            d.weekday() < 5
            and not non_working(d)
            and non_working(prev)
            and non_working(nxt)
            and any(x in stat or x in extra for x in (prev, nxt))
        )
        rows.append(
            {
                "date": ts,
                "holiday": int(is_holiday(d)),
                "holiday_statutory": int(d in stat),
                "holiday_eve": int(nxt in stat and d not in stat),
                "holiday_after": int(prev in stat and d not in stat),
                "bridge": int(bridge),
                "dst_day": int(_is_dst_day(d, cfg.timezone)),
            }
        )
    return pd.DataFrame(rows).set_index("date")


def _is_dst_day(d: dt.date, tz: str | None) -> bool:
    if tz is None:
        return False
    zone = ZoneInfo(tz)
    start = dt.datetime(d.year, d.month, d.day, tzinfo=zone)
    end = start + dt.timedelta(days=1)
    return start.utcoffset() != end.utcoffset()

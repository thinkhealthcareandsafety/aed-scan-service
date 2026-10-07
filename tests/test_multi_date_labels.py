"""Labels carry more than one date. These pin down the regression where the
manufacture date — the first date on a Philips battery label — was treated as
the only date in the transcription, vetoing correct expiry readings."""
from app.utils.date_parser import parse_all_expiry_dates, parse_expiry_date
from app.utils.validators import expiry_cross_check_agrees

PHILIPS_BATTERY = "2026-01-13 (manufacture), 2031-11-30 (install-before)"


def test_all_dates_are_found_in_order():
    assert parse_all_expiry_dates(PHILIPS_BATTERY) == ["2026-01-13", "2031-11-30"]


def test_full_date_is_not_also_read_as_its_leading_year_month():
    # "2031-11-30" must yield one date, not "2031-11-30" plus "2031-11".
    assert parse_all_expiry_dates("install-before 2031-11-30") == ["2031-11-30"]


def test_correct_expiry_agrees_even_when_it_is_not_the_first_date():
    # The exact production failure: right answer, vetoed as "implausible".
    assert expiry_cross_check_agrees("2031-11-30", PHILIPS_BATTERY) is True


def test_month_only_expiry_agrees_with_a_full_date_in_the_raw_text():
    assert expiry_cross_check_agrees("2031-11", PHILIPS_BATTERY) is True


def test_a_date_absent_from_the_label_is_still_rejected():
    # The cross-check must keep catching genuine misreads.
    assert expiry_cross_check_agrees("2029-04", PHILIPS_BATTERY) is False


def test_single_date_labels_behave_as_before():
    assert parse_all_expiry_dates("USE BY 2027-03") == ["2027-03"]
    assert expiry_cross_check_agrees("2027-03", "USE BY 2027-03") is True
    assert parse_expiry_date("2027-03-14") == "2027-03-14"


def test_empty_and_dateless_text():
    assert parse_all_expiry_dates("") == []
    assert parse_all_expiry_dates("no dates here") == []
    assert expiry_cross_check_agrees("2027-03", "no dates here") is True

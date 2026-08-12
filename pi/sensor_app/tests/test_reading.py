from core.reading import parse_device_range, parse_reference, read_anchored


def test_read_anchored_anchors_to_last_good():
    field = {
        "type": "temp", "decimals": 1, "int_digits": 2,
        "range_min": -50, "range_max": 100, "ref_value": 24.0, "last_good": 24.0,
    }
    value, valid, reason = read_anchored("2492", field)
    assert value == "24.9"
    assert valid is True
    assert reason == ""
    assert field["last_good"] == 24.9


def test_read_anchored_negative():
    field = {
        "type": "temp", "decimals": 1, "int_digits": 2,
        "range_min": -40, "range_max": 40, "ref_value": -18.0, "last_good": -18.0,
    }
    value, valid, reason = read_anchored("-185", field)
    assert valid is True
    assert value.startswith("-")


def test_read_anchored_clock_3_digits():
    field = {"type": "clock", "decimals": 0, "int_digits": 0}
    value, valid, reason = read_anchored("801", field)
    assert value == "8:01"
    assert valid is True


def test_read_anchored_clock_4_digits():
    field = {"type": "clock", "decimals": 0, "int_digits": 0}
    value, valid, reason = read_anchored("1234", field)
    assert value == "12:34"
    assert valid is True


def test_read_anchored_clock_too_few_digits():
    field = {"type": "clock", "decimals": 0, "int_digits": 0}
    value, valid, reason = read_anchored("12", field)
    assert valid is False
    assert "too few digits" in reason


def test_read_anchored_too_few_digits():
    field = {
        "type": "temp", "decimals": 1, "int_digits": 2,
        "range_min": -50, "range_max": 100, "ref_value": 24.0, "last_good": 24.0,
    }
    value, valid, reason = read_anchored("12", field)
    assert value is None
    assert valid is False
    assert "too few digits" in reason


def test_read_anchored_out_of_range():
    field = {
        "type": "temp", "decimals": 1, "int_digits": 2,
        "range_min": 0, "range_max": 10, "ref_value": 5.0, "last_good": 5.0,
    }
    # all 3-digit windows of "999" -> 99.9, out of [0,10]
    value, valid, reason = read_anchored("999", field)
    assert valid is False
    assert reason == "out of range"


def test_parse_device_range():
    lo, hi, raw = parse_device_range(["-50~+70C"])
    assert (lo, hi) == (-50, 70)
    assert raw == "-50~+70C"


def test_parse_device_range_no_match():
    lo, hi, raw = parse_device_range(["no range here"])
    assert (lo, hi, raw) == (None, None, None)


def test_parse_reference_decimal():
    val, intp, frac = parse_reference("25.3C")
    assert val == 25.3
    assert intp == 2
    assert frac == 1


def test_parse_reference_integer_fallback():
    val, intp, frac = parse_reference("not a number", "48%")
    assert val == 48.0
    assert intp == 2
    assert frac == 0


def test_parse_reference_none():
    val, intp, frac = parse_reference("", None)
    assert (val, intp, frac) == (None, None, None)

from core.reading import TemporalVoter


def test_voter_publishes_after_n_agree():
    v = TemporalVoter(k=5, n_agree=3)
    assert v.commit(24.9) == (None, False)
    assert v.commit(24.9) == (None, False)
    stable, is_stable = v.commit(24.9)
    assert is_stable is True
    assert stable == 24.9


def test_voter_flapping_holds_previous_stable():
    v = TemporalVoter(k=5, n_agree=3)
    v.commit(24.9)
    v.commit(24.9)
    stable, is_stable = v.commit(24.9)
    assert (stable, is_stable) == (24.9, True)

    # flapping noise afterwards should not move the published value
    stable, is_stable = v.commit(99.9)
    assert is_stable is False
    assert stable == 24.9

    stable, is_stable = v.commit(0.1)
    assert is_stable is False
    assert stable == 24.9


def test_voter_epsilon_tolerance():
    v = TemporalVoter(k=5, n_agree=3, epsilon=0.05)
    v.commit(24.90)
    v.commit(24.91)
    stable, is_stable = v.commit(24.89)
    assert is_stable is True


def test_voter_clock_exact_match_required():
    v = TemporalVoter(k=5, n_agree=3, is_clock=True)
    v.commit("12:34")
    v.commit("12:35")  # one off -- shouldn't agree even though "close"
    stable, is_stable = v.commit("12:34")
    assert is_stable is False
    assert stable is None

    v.commit("12:34")
    v.commit("12:34")
    stable, is_stable = v.commit("12:34")
    assert is_stable is True
    assert stable == "12:34"


def test_voter_reset():
    v = TemporalVoter(k=5, n_agree=3)
    v.commit(1.0)
    v.commit(1.0)
    v.commit(1.0)
    assert v._stable_value == 1.0
    v.reset()
    assert v._stable_value is None
    stable, is_stable = v.commit(5.0)
    assert is_stable is False

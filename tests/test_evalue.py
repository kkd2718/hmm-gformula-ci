"""Smoke tests for E-value computation (VanderWeele & Ding 2017)."""
import math
import pytest

from src.sensitivity.evalue import evalue_for_rr, evalue_for_rd


def test_evalue_rr_at_one():
    assert evalue_for_rr(1.0) == pytest.approx(1.0)


def test_evalue_rr_known_case():
    # VanderWeele & Ding 2017: RR=2 => E-value = 2 + sqrt(2) ~= 3.414
    assert evalue_for_rr(2.0) == pytest.approx(2.0 + math.sqrt(2.0), rel=1e-6)


def test_evalue_rr_below_one_uses_inverse():
    assert evalue_for_rr(0.5) == pytest.approx(evalue_for_rr(2.0), rel=1e-6)


def test_evalue_rd_via_rr():
    # p_treated=0.20, p_control=0.10 => RR=2 => same E-value
    assert evalue_for_rd(0.20, 0.10) == pytest.approx(evalue_for_rr(2.0), rel=1e-6)


def test_evalue_rejects_invalid():
    with pytest.raises(ValueError):
        evalue_for_rr(0.0)
    with pytest.raises(ValueError):
        evalue_for_rd(0.1, 0.0)

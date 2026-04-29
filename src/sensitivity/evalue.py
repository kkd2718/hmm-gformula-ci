"""E-value for unmeasured confounding sensitivity (VanderWeele & Ding 2017)."""
from __future__ import annotations
import math


def evalue_for_rr(rr: float) -> float:
    """E-value for a risk ratio (rr > 1 implies harmful exposure).

    Returns the minimum strength of unmeasured confounding (on the RR scale)
    that, jointly with the exposure-outcome association, must be present to
    explain the observed RR away (VanderWeele & Ding 2017 Ann Intern Med).
    """
    if rr <= 0:
        raise ValueError("rr must be positive.")
    rr_eff = rr if rr >= 1.0 else 1.0 / rr
    return rr_eff + math.sqrt(rr_eff * (rr_eff - 1.0))


def evalue_for_rd(p_treated: float, p_control: float) -> float:
    """E-value for a risk difference, computed via the implied risk ratio.

    Converts (p_treated, p_control) to RR = p_treated / p_control and
    delegates to evalue_for_rr.
    """
    if p_control <= 0 or p_treated <= 0:
        raise ValueError("Risks must be strictly positive.")
    return evalue_for_rr(p_treated / p_control)

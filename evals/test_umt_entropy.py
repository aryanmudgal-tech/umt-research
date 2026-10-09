"""Offline tests for the fatigue entropy model (tools/umt/entropy.py).

The model is Lee & Basaran, Mech. Mater. 164 (2022) 104116. The tests hold it
to that paper's own numbers where the paper gives them, and to a slower
independent calculation where it does not. No API calls.
"""

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.umt import entropy as en  # noqa: E402
from tools.umt.entropy import DP600  # noqa: E402

MPa = 1e6
# Section 4 of the paper: "TSI reaches 1 around 8.5e4 cycles under 325 MPa ...
# 1.4e5 cycles [at 300 MPa] ... around 3.6e5 cycles [at 275 MPa]".
PAPER_LIVES = {325 * MPa: 8.5e4, 300 * MPa: 1.4e5, 275 * MPa: 3.6e5}


# ---------------------------------------------------------------- the material


def test_the_localisation_constants_are_the_papers_equations_20_and_21():
    assert DP600.shear_modulus == pytest.approx(80.77e9, rel=1e-4)
    assert DP600.bulk_modulus == pytest.approx(175e9, rel=1e-4)
    # b = 6 (K + 2 mu) / (5 (3K + 4 mu)), then C = 2 mu (1 - b)(1 - f_v)
    assert DP600.localisation_modulus == pytest.approx(76.15e9, rel=1e-4)
    assert DP600.plastic_modulus == pytest.approx(1.5 * 76.15e9 + 1e9, rel=1e-4)


# ------------------------------------------------------------ work per cycle


@pytest.mark.parametrize("amplitude", [262 * MPa, 275 * MPa, 300 * MPa, 325 * MPa, 400 * MPa])
def test_the_closed_loop_work_is_what_the_papers_incremental_scheme_gives(amplitude):
    closed = en.plastic_work_per_cycle(DP600, amplitude)
    stepped = en.plastic_work_by_return_mapping(DP600, amplitude, steps=4000)
    assert closed == pytest.approx(stepped, rel=2e-3)


@pytest.mark.parametrize("amplitude", [0.0, 100 * MPa, 259.9 * MPa, 260 * MPa])
def test_no_work_is_dissipated_at_or_below_the_micro_yield_stress(amplitude):
    assert en.plastic_work_per_cycle(DP600, amplitude) == 0.0
    assert en.plastic_work_by_return_mapping(DP600, amplitude) == 0.0


def test_the_stabilised_loop_at_300_mpa_matches_the_papers_figure_11():
    """Fig. 11 marks reverse yield at a micro strain of -0.868e-3."""
    plastic_amplitude = (300 * MPa - DP600.micro_yield) / DP600.plastic_modulus
    macro_stress = -DP600.micro_yield + DP600.plastic_modulus * plastic_amplitude  # reverse yield
    b = 1 - DP600.localisation_modulus / (2 * DP600.shear_modulus * (1 - DP600.defect_fraction))
    elastic = macro_stress / DP600.elastic_modulus - (1 - b) * (1 - DP600.defect_fraction) * plastic_amplitude
    assert elastic + plastic_amplitude == pytest.approx(-0.868e-3, rel=0.01)


def test_tension_and_compression_amplitudes_dissipate_the_same():
    assert en.plastic_work_per_cycle(DP600, -300 * MPa) == en.plastic_work_per_cycle(DP600, 300 * MPa)


# ------------------------------------------------------- the state index


def test_the_state_index_is_equation_5():
    ds = 100.0  # J/(kg K)
    expected = 1 - math.exp(-ds * DP600.molar_mass / en.GAS_CONSTANT)
    assert en.state_index(DP600, ds * DP600.density) == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(0.4892, abs=1e-4)


def test_entropy_generated_inverts_the_state_index():
    for index in (1e-3, 0.3, 0.995):
        entropy = en.entropy_generated(DP600, index, initial=1e-4)
        assert en.state_index(DP600, entropy, initial=1e-4) == pytest.approx(index, rel=1e-12)
    assert en.state_index(DP600, 0.0, initial=0.25) == 0.25


def test_a_state_index_of_zero_never_moves_so_it_is_refused():
    with pytest.raises(ValueError, match="never moves"):
        en.simulate(DP600, 300 * MPa, initial=0.0)


# ------------------------------------------------------------ fatigue life


@pytest.mark.parametrize("amplitude", sorted(PAPER_LIVES))
def test_the_lives_are_the_papers(amplitude):
    """One number is fitted (the starting TSI, at 300 MPa); the other two lives
    are predictions and have to land on the paper's Fig. 16."""
    life = en.simulate(DP600, amplitude)["cycles_to_failure"]
    assert life == pytest.approx(PAPER_LIVES[amplitude], rel=0.04)


def test_the_fitted_seed_is_fitted_at_300_mpa():
    assert en.simulate(DP600, 300 * MPa)["cycles_to_failure"] == pytest.approx(1.4e5, rel=1e-3)


@pytest.mark.parametrize("amplitude", sorted(PAPER_LIVES))
def test_blocks_agree_with_adding_it_up_one_cycle_at_a_time(amplitude):
    blocks = en.simulate(DP600, amplitude)["cycles_to_failure"]
    assert blocks == pytest.approx(en.simulate_cycle_by_cycle(DP600, amplitude), rel=5e-4)


def test_the_life_does_not_depend_on_the_block_size():
    coarse = en.simulate(DP600, 300 * MPa, max_step=0.05)["cycles_to_failure"]
    fine = en.simulate(DP600, 300 * MPa, max_step=0.002)["cycles_to_failure"]
    assert coarse == pytest.approx(fine, rel=1e-3)


def test_below_the_endurance_limit_is_a_run_out():
    out = en.simulate(DP600, 250 * MPa)
    assert out["cycles_to_failure"] is None
    assert out["work_per_cycle"] == 0.0
    assert en.simulate_cycle_by_cycle(DP600, 250 * MPa) is None
    assert en.isothermal_life(DP600, 250 * MPa) == math.inf


def test_running_out_of_cycles_is_a_run_out_not_a_failure():
    out = en.simulate(DP600, 275 * MPa, max_cycles=1e5)
    assert out["cycles_to_failure"] is None
    assert out["history"][-1]["cycles"] == pytest.approx(1e5)
    assert out["history"][-1]["index"] < en.PAPER_CRITICAL


def test_the_entropy_at_failure_does_not_depend_on_the_stress_amplitude():
    """The fatigue fracture entropy of the paper's Fig. 14."""
    values = [en.simulate(DP600, a)["entropy_at_failure"] for a in sorted(PAPER_LIVES)]
    assert max(values) == pytest.approx(min(values), rel=1e-12)
    assert values[0] == pytest.approx(6.2e6, rel=0.01)  # J/(K m^3) at a TSI of 0.995


def test_the_state_index_and_the_temperature_only_ever_rise_under_load():
    history = en.simulate(DP600, 300 * MPa)["history"]
    for before, after in zip(history, history[1:]):
        assert after["cycles"] > before["cycles"]
        assert after["index"] > before["index"]
        assert after["temperature"] >= before["temperature"] - 1e-9
    assert history[0]["index"] == en.PAPER_SEED
    assert history[-1]["index"] == en.PAPER_CRITICAL


def test_the_odds_of_the_state_index_grow_exponentially_at_constant_temperature():
    work = en.plastic_work_per_cycle(DP600, 300 * MPa)
    rate = DP600.entropy_scale * DP600.defect_fraction * work / 300.0
    cold = en.FatigueMaterial(**{**DP600.__dict__, "specific_heat": 1e12})  # no self-heating
    index, _ = en.advance(cold, 1e-3, 300.0, work, 30.0, 5e4, 300.0)
    assert index / (1 - index) == pytest.approx(1e-3 / (1 - 1e-3) * math.exp(rate * 5e4), rel=1e-6)
    assert en.isothermal_life(DP600, 300 * MPa, temperature=300.0) == pytest.approx(
        math.log((0.995 / 0.005) / (en.PAPER_SEED / (1 - en.PAPER_SEED))) / rate, rel=1e-12
    )


def test_a_hotter_specimen_lives_longer_because_entropy_is_work_over_temperature():
    assert en.isothermal_life(DP600, 300 * MPa, temperature=350.0) > en.isothermal_life(
        DP600, 300 * MPa, temperature=300.0
    )


def test_the_temperature_settles_where_heating_balances_the_thermal_exchange():
    work = en.plastic_work_per_cycle(DP600, 300 * MPa)
    steady = en.steady_temperature(DP600, 1.0, work, 30.0, 300.0)
    _, temperature = en.advance(DP600, 0.999999, 300.0, work, 30.0, 30.0 * 80.0 * 20, 300.0)
    assert temperature == pytest.approx(steady, rel=1e-4)
    # unloaded, it relaxes back to ambient over the thermal time
    _, cooled = en.advance(DP600, 0.5, 320.0, 0.0, 30.0, 30.0 * 80.0, 300.0)
    assert cooled == pytest.approx(300.0 + 20.0 / math.e, rel=1e-9)

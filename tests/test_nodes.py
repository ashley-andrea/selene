"""
Unit tests — tests individual node logic using Level 1 static mocks.

These tests use pytest's patch decorator to replace the real API clients
with static mock functions. No network calls are made.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tests.mocks import cluster_api as mock_cluster
from tests.mocks import simulator_api as mock_simulator


# ── Test State Factory ──────────────────────────────────────────────────────


def make_state(**overrides) -> dict:
    """Factory for a minimal valid SystemState for testing."""
    base = {
        "patient_data": {
            "age": 28,
            "pathologies": [],
            "habits": [],
            "medical_history": [],
        },
        "cluster_profile": "cluster_2",
        "cluster_confidence": 0.88,
        "relative_risk_rules": [],
        "candidate_pool": ["pill_a", "pill_b", "pill_c"],
        "simulated_results": {},
        "utility_scores": {},
        "best_candidate": None,
        "iteration": 0,
        "converged": False,
        "previous_best_utility": None,
        "reason_codes": [],
    }
    base.update(overrides)
    return base


# ── Validator Tests ─────────────────────────────────────────────────────────


class TestValidator:
    def test_valid_patient_passes(self):
        from agent.nodes import validator

        state = make_state()
        result = validator.run(state)
        assert result["patient_data"]["age"] == 28
        assert result["converged"] is False
        assert result["iteration"] == 0

    def test_missing_field_raises(self):
        from agent.nodes import validator

        state = make_state(patient_data={"age": 28})
        with pytest.raises(ValueError, match="Missing required patient field"):
            validator.run(state)

    def test_age_out_of_range_raises(self):
        from agent.nodes import validator

        state = make_state(
            patient_data={
                "age": 10,
                "pathologies": [],
                "habits": [],
                "medical_history": [],
            }
        )
        with pytest.raises(ValueError, match="must be between"):
            validator.run(state)

    def test_normalizes_to_lowercase(self):
        from agent.nodes import validator

        state = make_state(
            patient_data={
                "age": 30,
                "pathologies": ["Hypertension", " MIGRAINES "],
                "habits": ["SMOKING"],
                "medical_history": ["DVT"],
            }
        )
        result = validator.run(state)
        assert result["patient_data"]["pathologies"] == ["hypertension", "migraines"]
        assert result["patient_data"]["habits"] == ["smoking"]
        assert result["patient_data"]["medical_history"] == ["dvt"]


# ── Cluster Tests ───────────────────────────────────────────────────────────


class TestCluster:
    @patch(
        "agent.nodes.cluster.call_cluster_model",
        mock_cluster.call_cluster_model,
    )
    def test_high_confidence_returns_string_profile(self):
        from agent.nodes import cluster

        state = make_state(patient_data={"age": 28, "pathologies": [], "habits": [], "medical_history": []})
        result = cluster.run(state)
        assert result["cluster_confidence"] >= 0.70
        assert isinstance(result["cluster_profile"], str)

    @patch(
        "agent.nodes.cluster.call_cluster_model",
        mock_cluster.call_cluster_model,
    )
    def test_low_confidence_triggers_adjustment(self):
        """Age 50 triggers confidence < 0.70 in the mock."""
        from agent.nodes import cluster

        # Mock the LLM to return a valid JSON adjustment
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({"weight_adjustment": 1.2, "rationale": "test"})
        )

        state = make_state(
            patient_data={"age": 50, "pathologies": ["diabetes"], "habits": [], "medical_history": []}
        )

        with patch("agent.nodes.cluster.get_llm", return_value=mock_llm):
            result = cluster.run(state)

        assert result["cluster_confidence"] < 0.70
        assert isinstance(result["cluster_profile"], dict)
        assert result["cluster_profile"]["low_confidence"] is True
        assert result["cluster_profile"]["weight_adjustment"] == 1.2


# ── Candidate Generation Tests ──────────────────────────────────────────────


class TestCandidateGen:
    def test_generates_candidates_for_healthy_patient(self):
        from agent.nodes import candidate_gen

        state = make_state(
            patient_data={"age": 25, "pathologies": [], "habits": [], "medical_history": []},
            cluster_profile="cluster_0",
        )
        result = candidate_gen.run(state)
        assert len(result["candidate_pool"]) > 0
        assert isinstance(result["relative_risk_rules"], list)

    def test_excludes_combined_pills_for_vte_history(self):
        from agent.nodes import candidate_gen
        from agent.pill_database import get_pill_by_id

        state = make_state(
            patient_data={
                "age": 35,
                "pathologies": [],
                "habits": [],
                "medical_history": ["dvt"],
            },
            cluster_profile="cluster_2",
        )
        result = candidate_gen.run(state)
        # All remaining pills should be progestin-only (no ethinyl estradiol)
        for pill_id in result["candidate_pool"]:
            pill = get_pill_by_id(pill_id)
            assert "ETHINYL ESTRADIOL" not in pill.get("substance_name", "").upper(), (
                f"{pill_id} should have been excluded (combined + VTE history)"
            )

    def test_smoking_over_35_excludes_combined(self):
        from agent.nodes import candidate_gen
        from agent.pill_database import get_pill_by_id

        state = make_state(
            patient_data={
                "age": 40,
                "pathologies": [],
                "habits": ["smoking"],
                "medical_history": [],
            },
            cluster_profile="cluster_3",
        )
        result = candidate_gen.run(state)
        for pill_id in result["candidate_pool"]:
            pill = get_pill_by_id(pill_id)
            assert "ETHINYL ESTRADIOL" not in pill.get("substance_name", "").upper(), (
                f"{pill_id} should have been excluded (smoker over 35)"
            )


# ── Utility Tests ───────────────────────────────────────────────────────────


class TestUtility:
    def test_selects_best_candidate(self):
        from agent.nodes import utility

        state = make_state(
            simulated_results={
                "pill_a": mock_simulator.FAKE_SIMULATION_RESULTS["pill_a"],
                "pill_b": mock_simulator.FAKE_SIMULATION_RESULTS["pill_b"],
                "pill_c": mock_simulator.FAKE_SIMULATION_RESULTS["pill_c"],
            }
        )
        result = utility.run(state)
        # pill_a has the best simulated profile — should always win
        assert result["best_candidate"] == "pill_a"
        assert result["utility_scores"]["pill_a"] > result["utility_scores"]["pill_b"]
        assert result["utility_scores"]["pill_b"] > result["utility_scores"]["pill_c"]

    def test_empty_results_returns_none(self):
        from agent.nodes import utility

        state = make_state(simulated_results={})
        result = utility.run(state)
        assert result["best_candidate"] is None
        assert result["utility_scores"] == {}

    def test_utility_formula_correctness(self):
        from agent.nodes.utility import DEFAULT_WEIGHTS, compute_utility

        sim = {
            "severe_event_probability": 0.01,
            "discontinuation_probability": 0.10,
            "mild_side_effect_score": 0.30,
            "contraceptive_effectiveness": 0.95,
        }
        expected = (
            -DEFAULT_WEIGHTS["alpha"] * 0.01 
            - DEFAULT_WEIGHTS["beta"] * 0.10 
            - DEFAULT_WEIGHTS["gamma"] * 0.30 
            + DEFAULT_WEIGHTS["delta"] * 0.95
        )
        assert abs(compute_utility(sim, DEFAULT_WEIGHTS) - expected) < 1e-9


# ── Convergence Tests ───────────────────────────────────────────────────────


class TestConvergence:
    def test_converges_after_max_iterations(self):
        from agent.nodes import convergence

        state = make_state(
            iteration=5,  # MAX_ITERATIONS reached
            utility_scores={"pill_a": 0.85},
            best_candidate="pill_a",
            simulated_results={"pill_a": mock_simulator.FAKE_SIMULATION_RESULTS["pill_a"]},
            previous_best_utility=0.84,
        )

        # Mock the LLM for reason code generation
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({
                "converged": True,
                "confidence": 0.9,
                "medical_rationale": "Maximum iterations reached with stable utility",
                "reason_codes": ["reason 1", "reason 2"],
            })
        )

        with patch("agent.nodes.convergence.get_llm", return_value=mock_llm):
            result = convergence.run(state)

        assert result["converged"] is True
        assert len(result["reason_codes"]) >= 2

    def test_does_not_converge_on_first_iteration(self):
        from agent.nodes import convergence

        state = make_state(
            iteration=0,
            utility_scores={"pill_a": 0.85},
            best_candidate="pill_a",
            previous_best_utility=None,
        )
        result = convergence.run(state)
        assert result["converged"] is False
        assert result["iteration"] == 1

    def test_converges_on_small_improvement(self):
        from agent.nodes import convergence

        state = make_state(
            iteration=2,
            utility_scores={"pill_a": 0.850},
            best_candidate="pill_a",
            simulated_results={"pill_a": mock_simulator.FAKE_SIMULATION_RESULTS["pill_a"]},
            previous_best_utility=0.849,  # Improvement < EPSILON
        )

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps(["stable result", "minimal change"])
        )

        with patch("agent.nodes.convergence.get_llm", return_value=mock_llm):
            result = convergence.run(state)

        assert result["converged"] is True


# ── Safe Gate Tests ─────────────────────────────────────────────────────────


class TestSafeGate:
    def test_healthy_patient_gets_all_pills(self):
        from agent.safe_gate import apply_safe_gate

        result = apply_safe_gate(
            {"age": 25, "pathologies": [], "habits": [], "medical_history": []},
            "cluster_0",
        )
        # All 296 pills should be available for healthy patient
        assert len(result["candidate_pool"]) == 296

    def test_breast_cancer_excludes_all(self):
        from agent.safe_gate import apply_safe_gate

        result = apply_safe_gate(
            {"age": 30, "pathologies": [], "habits": [], "medical_history": ["breast_cancer"]},
            "cluster_2",
        )
        # Breast cancer excludes all pills (both combined and progestin-only)
        assert len(result["candidate_pool"]) == 0

    def test_dvt_excludes_combined_only(self):
        from agent import pill_database
        from agent.safe_gate import apply_safe_gate

        result = apply_safe_gate(
            {"age": 30, "pathologies": [], "habits": [], "medical_history": ["dvt"]},
            "cluster_2",
        )
        # Only progestin-only pills should remain (no ethinyl estradiol)
        for pill_id in result["candidate_pool"]:
            pill = pill_database.get_pill_by_id(pill_id)
            assert "ETHINYL ESTRADIOL" not in pill.get("substance_name", "").upper()

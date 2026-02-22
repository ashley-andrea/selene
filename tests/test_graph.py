"""
Integration tests — tests the full agent graph using the Level 2 mock server.

These tests assume the mock server is running on port 8001.
In CI, start it via the mock_server pytest fixture below.

These tests also require an LLM provider to be configured (or mocked).
"""

import json
import os
import subprocess
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest


# ── Mock Server Fixture ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def mock_server():
    """Start the mock ML model server for integration tests."""
    proc = subprocess.Popen(
        ["python", "-m", "tests.mock_server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(2.0)  # Give it time to start

    # Verify it's running
    try:
        resp = httpx.get("http://localhost:8001/health", timeout=5.0)
        assert resp.status_code == 200
    except Exception:
        proc.terminate()
        pytest.skip("Mock server failed to start")

    yield proc
    proc.terminate()
    proc.wait()


@pytest.fixture(autouse=True)
def set_env_vars():
    """Set environment variables pointing to the mock server."""
    env_overrides = {
        "CLUSTER_API_URL": "http://localhost:8001/cluster/predict",
        "SIMULATOR_API_URL": "http://localhost:8001/simulator/simulate",
    }
    original = {k: os.environ.get(k) for k in env_overrides}
    os.environ.update(env_overrides)
    yield
    for k, v in original.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ── Mock LLM for Integration Tests ─────────────────────────────────────────


def _make_mock_llm():
    """Create a mock LLM that returns valid JSON for all agent prompts."""
    mock_llm = MagicMock()

    def mock_invoke(messages):
        content = messages[0].content if messages else ""

        # Weight adjustment prompt
        if "weight_adjustment" in content.lower():
            return MagicMock(
                content=json.dumps({"weight_adjustment": 1.1, "rationale": "mock adjustment"})
            )

        # Reason codes prompt
        if "reason codes" in content.lower() or "reason strings" in content.lower():
            return MagicMock(
                content=json.dumps([
                    "lowest risk in cluster",
                    "high predicted adherence",
                    "matches patient profile",
                ])
            )

        # Default: return a safe JSON response
        return MagicMock(content=json.dumps({"result": "ok"}))

    mock_llm.invoke = mock_invoke
    return mock_llm


# ── Graph Integration Tests ─────────────────────────────────────────────────


class TestGraphIntegration:
    """Tests the full LangGraph agent loop with mock server + mock LLM."""

    def test_full_recommendation_standard_patient(self, mock_server):
        from agent.graph import build_graph

        mock_llm = _make_mock_llm()

        with patch("agent.nodes.cluster.get_llm", return_value=mock_llm), \
             patch("agent.nodes.convergence.get_llm", return_value=mock_llm):

            graph = build_graph()
            initial_state = {
                "patient_data": {
                    "age": 28,
                    "pathologies": [],
                    "habits": ["smoking"],
                    "medical_history": [],
                },
                "iteration": 0,
                "converged": False,
                "cluster_profile": None,
                "cluster_confidence": None,
                "relative_risk_rules": [],
                "candidate_pool": [],
                "simulated_results": {},
                "utility_scores": {},
                "best_candidate": None,
                "previous_best_utility": None,
                "reason_codes": [],
            }

            final_state = graph.invoke(initial_state)

        assert final_state["converged"] is True
        assert final_state["best_candidate"] is not None
        assert len(final_state["utility_scores"]) > 0
        assert final_state["iteration"] > 0

    def test_high_risk_patient_gets_limited_pool(self, mock_server):
        from agent.graph import build_graph

        mock_llm = _make_mock_llm()

        with patch("agent.nodes.cluster.get_llm", return_value=mock_llm), \
             patch("agent.nodes.convergence.get_llm", return_value=mock_llm):

            graph = build_graph()
            initial_state = {
                "patient_data": {
                    "age": 42,
                    "pathologies": ["hypertension", "migraines"],
                    "habits": [],
                    "medical_history": ["dvt"],
                },
                "iteration": 0,
                "converged": False,
                "cluster_profile": None,
                "cluster_confidence": None,
                "relative_risk_rules": [],
                "candidate_pool": [],
                "simulated_results": {},
                "utility_scores": {},
                "best_candidate": None,
                "previous_best_utility": None,
                "reason_codes": [],
            }

            final_state = graph.invoke(initial_state)

        assert final_state["converged"] is True
        assert final_state["best_candidate"] is not None
        # High-risk patient with DVT history should have fewer pills than all 9 available (combined pills excluded)
        assert len(final_state["candidate_pool"]) < 9

    def test_recommendation_converges_within_max_iterations(self, mock_server):
        from agent.nodes.convergence import MAX_ITERATIONS
        from agent.graph import build_graph

        mock_llm = _make_mock_llm()

        with patch("agent.nodes.cluster.get_llm", return_value=mock_llm), \
             patch("agent.nodes.convergence.get_llm", return_value=mock_llm):

            graph = build_graph()
            initial_state = {
                "patient_data": {
                    "age": 30,
                    "pathologies": ["pcos"],
                    "habits": [],
                    "medical_history": [],
                },
                "iteration": 0,
                "converged": False,
                "cluster_profile": None,
                "cluster_confidence": None,
                "relative_risk_rules": [],
                "candidate_pool": [],
                "simulated_results": {},
                "utility_scores": {},
                "best_candidate": None,
                "previous_best_utility": None,
                "reason_codes": [],
            }

            final_state = graph.invoke(initial_state)

        assert final_state["converged"] is True
        assert final_state["iteration"] <= MAX_ITERATIONS + 1


# ── Mock Server Endpoint Tests ──────────────────────────────────────────────


class TestMockServerEndpoints:
    """Direct HTTP tests against the mock server endpoints."""

    def test_cluster_predict(self, mock_server):
        response = httpx.post(
            "http://localhost:8001/cluster/predict",
            json={"patient": {"age": 28, "pathologies": [], "habits": [], "medical_history": []}},
        )
        assert response.status_code == 200
        data = response.json()
        assert "cluster_profile" in data
        assert "cluster_confidence" in data
        assert 0.0 <= data["cluster_confidence"] <= 1.0

    def test_simulator_simulate(self, mock_server):
        response = httpx.post(
            "http://localhost:8001/simulator/simulate",
            json={
                "candidate_pill": {
                    "pill_id": "pill_levonorgestrel_30",
                    "set_id": "test_pill_id",
                    "brand_name": "Test Pill",
                    "substance_name": "LEVONORGESTREL"
                },
                "patient": {"age": 28, "pathologies": [], "habits": [], "medical_history": []},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "discontinuation_probability" in data
        assert "severe_event_probability" in data
        assert "mild_side_effect_score" in data
        assert "contraceptive_effectiveness" in data

    def test_health_check(self, mock_server):
        response = httpx.get("http://localhost:8001/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

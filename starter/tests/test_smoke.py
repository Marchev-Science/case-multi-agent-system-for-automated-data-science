"""Smoke tests — the scaffold imports and runs (it just doesn't do anything yet).

These are intentionally trivial. Once your agents do real work, replace these
with meaningful tests of your orchestrator and agent contracts.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from maads.agents import DeveloperAgent
from maads.config import load_case_config
from maads.data_utils import CASE_SHORTHANDS
from maads.state import (
    CrispDMState,
    ModelRun,
    Phase,
    SUBSTEPS,
    SUBSTEP_NAMES,
    SUBSTEP_OWNER,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_all_three_configs_load():
    for case in ("titanic", "house_prices", "disaster_tweets"):
        cfg = load_case_config(REPO_ROOT / "configs" / f"{case}.yaml")
        assert cfg.case_id == case
        assert cfg.problem_statement
        assert cfg.target_column


def test_state_initialises_from_config():
    cfg = load_case_config(REPO_ROOT / "configs" / "titanic.yaml")
    state = CrispDMState.from_config(cfg)
    assert state.case_id == "titanic"
    assert state.phase == Phase.BUSINESS_UNDERSTANDING
    assert state.substep == "1.1"
    assert state.log == []
    assert state.loop_history == []
    # All six phase sub-objects exist and are empty.
    assert state.bu.business_objectives is None
    assert state.du.data_description_report is None
    assert state.dp.dataset == {}
    assert state.md.models == []
    assert state.ev.decision is None
    assert state.dep.submission_path is None


def test_24_substeps_with_names_and_owners():
    """The CRISP-DM 1.0 Reference Model has exactly 24 generic tasks."""
    assert sum(len(v) for v in SUBSTEPS.values()) == 24
    flat_ids = [s for v in SUBSTEPS.values() for s in v]
    assert set(flat_ids) == set(SUBSTEP_NAMES.keys()) == set(SUBSTEP_OWNER.keys())
    # Spot-check a few canonical names.
    assert SUBSTEP_NAMES["1.1"] == "Determine Business Objectives"
    assert SUBSTEP_NAMES["3.3"] == "Construct Data"
    assert SUBSTEP_NAMES["4.4"] == "Assess Model"
    assert SUBSTEP_NAMES["6.4"] == "Review Project"


def test_prereqs_block_phase_jumping():
    """The orchestrator must not dispatch a substep whose prereqs are unmet."""
    cfg = load_case_config(REPO_ROOT / "configs" / "titanic.yaml")
    state = CrispDMState.from_config(cfg)

    assert state.substep_prereqs_satisfied("4.1") is False
    state.dp.dataset = {"train": "x.parquet", "test": "y.parquet"}
    assert state.substep_prereqs_satisfied("4.1") is True

    assert state.substep_prereqs_satisfied("5.1") is False
    state.md.models.append(ModelRun(technique="logistic_regression", cv_score=0.78))
    assert state.substep_prereqs_satisfied("5.1") is True

    assert state.substep_prereqs_satisfied("6.1") is False
    state.md.chosen_model = state.md.models[0]
    assert state.substep_prereqs_satisfied("6.1") is True


def test_view_for_returns_only_agent_slice():
    """Token-economy check: view_for must NOT return the full state."""
    cfg = load_case_config(REPO_ROOT / "configs" / "titanic.yaml")
    state = CrispDMState.from_config(cfg)

    view = state.view_for("data_scientist")
    assert "case_id" in view
    assert "data_mining_goals" in view
    # The Data Scientist should not get the BU sub-object — that's for the
    # Domain Expert. The view should be a flat dict, not the state itself.
    assert "bu" not in view


def test_developer_has_debugging_api():
    """The Developer must expose its debugging-toolkit methods as part of
    its public API — they are stubs, but they must exist."""
    expected = {"classify_error", "propose_fix", "re_execute",
                "repair_json", "schema_check"}
    have = {m for m in dir(DeveloperAgent) if not m.startswith("_")}
    missing = expected - have
    assert not missing, f"DeveloperAgent missing debugging methods: {missing}"


def test_case_shorthands_are_just_a_convenience():
    """The data-download path must accept any Kaggle slug.

    The CASE_SHORTHANDS map is a convenience; the system itself is not
    limited to the keys in it.
    """
    assert set(CASE_SHORTHANDS) == {"titanic", "house_prices", "disaster_tweets"}
    # The download function used directly must take an arbitrary slug.
    from maads.data_utils import download_kaggle_competition
    import inspect
    sig = inspect.signature(download_kaggle_competition)
    assert "slug" in sig.parameters


@pytest.mark.skip(reason="Implement once orchestrator.run() exists.")
def test_full_pipeline_runs_on_titanic():
    pass  # TODO: end-to-end test on a tiny subset.

import numpy as np
import pytest

from src.agents.q_learning_agent import QLearningAgent


def _bins(**overrides):
    bins = {
        "belief_margin": 4,
        "uncertainty": 5,
        "engagement": 5,
        "budget": 4,
        "step": 6,
        "asked": 4,
        "recommended": 4,
        "repeats": 4,
    }
    bins.update(overrides)
    return bins


def test_q_learning_update_changes_value():
    agent = QLearningAgent(
        action_size=4,
        learning_rate=0.5,
        discount_factor=0.9,
        epsilon_start=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        bins=_bins(),
    )
    s = (0, 0, 0, 0, 0, 0, 0, 0)
    ns = (1, 0, 0, 0, 0, 0, 0, 0)
    agent.update(s, action=1, reward=1.0, next_state=ns, done=False)
    assert agent.q_table[s][1] > 0


def test_q_learning_state_discretization_is_stable():
    agent = QLearningAgent(
        action_size=3,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.99,
        bins=_bins(),
    )
    obs = np.array([0.1, 0.2, 0.4, 0.1, 0.2, 0.7, 0.2, 0.1, 0.0, 0.75, 0.8, 0.15])
    assert isinstance(agent.discretize_state(obs), tuple)


def test_q_learning_greedy_tie_break_is_deterministic_and_not_first_index():
    agent = QLearningAgent(
        action_size=4,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        bins=_bins(),
    )
    state = (1, 0, 0, 0, 0, 0, 0, 0)

    first = agent.select_action(state, explore=False)
    second = agent.select_action(state, explore=False)

    assert first == second
    assert first != 0


def test_q_learning_training_tie_break_uses_random_choice(monkeypatch: pytest.MonkeyPatch):
    agent = QLearningAgent(
        action_size=4,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        bins=_bins(),
    )
    state = (2, 1, 0, 0, 0, 0, 0, 0)

    monkeypatch.setattr(np.random, "choice", lambda values: values[-1])

    assert agent.select_action(state, explore=True) == 3


def test_q_learning_state_uses_belief_margin_and_repeats_bins():
    agent = QLearningAgent(
        action_size=3,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        bins=_bins(recommended=4, repeats=2, belief_margin=4),
    )
    confident_obs = np.array([0.55, 0.15, 0.1, 0.1, 0.1, 0.7, 0.2, 0.1, 0.6, 0.75, 0.8, 0.15])
    uncertain_obs = np.array([0.26, 0.24, 0.2, 0.15, 0.15, 0.7, 0.2, 0.1, 0.6, 0.75, 0.8, 0.15])

    confident_state = agent.discretize_state(confident_obs)
    uncertain_state = agent.discretize_state(uncertain_obs)

    assert confident_state[0] != uncertain_state[0]
    assert confident_state[-1] == 1


import numpy as np

from src.agents.q_learning_agent import QLearningAgent


def test_q_learning_update_changes_value():
    agent = QLearningAgent(
        action_size=4,
        learning_rate=0.5,
        discount_factor=0.9,
        epsilon_start=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        bins={
            "uncertainty": 5,
            "engagement": 5,
            "budget": 4,
            "step": 6,
            "asked": 4,
            "recommended": 4,
        },
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
        bins={
            "uncertainty": 5,
            "engagement": 5,
            "budget": 4,
            "step": 6,
            "asked": 4,
            "recommended": 4,
        },
    )
    obs = np.array([0.1, 0.2, 0.4, 0.1, 0.2, 0.7, 0.2, 0.1, 0.0, 0.75, 0.8, 0.15])
    assert isinstance(agent.discretize_state(obs), tuple)


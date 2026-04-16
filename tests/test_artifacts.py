import tempfile
from pathlib import Path

from src.agents.q_learning_agent import QLearningAgent


def test_q_table_save_load_roundtrip():
    agent = QLearningAgent(
        action_size=3,
        learning_rate=0.2,
        discount_factor=0.95,
        epsilon_start=0.8,
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
    state = (0, 0, 0, 0, 0, 0, 0, 0)
    agent.q_table[state] = agent._ensure_state(state)
    agent.q_table[state][0] = 1.23

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        agent.save(
            tmp_dir,
            metadata={
                "action_size": 3,
                "learning_rate": 0.2,
                "discount_factor": 0.95,
                "epsilon_min": 0.1,
                "epsilon_decay": 0.99,
            },
        )
        loaded = QLearningAgent.load(tmp_dir)
        assert loaded.q_table[state][0] == agent.q_table[state][0]


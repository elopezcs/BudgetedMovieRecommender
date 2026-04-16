import numpy as np

from src.env.user_simulator import GENRES, UserSimulator


def test_user_simulator_reset_outputs_valid_user_type():
    sim = UserSimulator()
    user_type = sim.reset(seed=42)
    assert user_type in {
        "action_focused",
        "balanced_viewer",
        "novelty_seeking",
        "question_sensitive",
    }
    assert np.isclose(sim.true_preferences.sum(), 1.0, atol=1e-6)
    assert len(sim.true_preferences) == len(GENRES)


def test_user_simulator_question_and_recommendation_outputs():
    sim = UserSimulator()
    sim.reset(seed=42, user_type="balanced_viewer")
    q_out = sim.respond_to_question("familiar", repeated=False)
    assert 0.0 <= q_out["abandon_prob"] <= 1.0
    assert 0 <= q_out["hinted_genre"] < len(GENRES)

    r_out = sim.respond_to_recommendation(genre_index=0, repeated=False)
    assert isinstance(r_out["accepted"], bool)
    assert 0.0 <= r_out["abandon_prob"] <= 1.0


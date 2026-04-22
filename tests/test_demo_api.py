from fastapi.testclient import TestClient

from src.demo_api.app import app


client = TestClient(app)


def test_auto_mode_replaces_blocked_ask_with_fallback_recommendation():
    start = client.post(
        "/api/demo/session/start",
        json={
            "policy": "always_ask",
            "user_profile": "balanced_viewer",
            "question_budget": 1,
            "seed": 123,
            "mode": "auto",
        },
    )
    assert start.status_code == 200
    session_id = start.json()["session_id"]

    first = client.post("/api/demo/session/next", json={"session_id": session_id})
    second = client.post("/api/demo/session/next", json={"session_id": session_id})

    assert first.status_code == 200
    assert second.status_code == 200

    first_step = first.json()["latest_step"]
    second_step = second.json()["latest_step"]

    assert first_step["action_type"] == "ask"
    assert first_step["question_asked"] is True
    assert first_step["over_budget"] is False

    assert second_step["action_type"] == "recommend"
    assert second_step["question_asked"] is False
    assert second_step["over_budget"] is False
    assert second_step["fallback_applied"] is True
    assert "question budget exhausted" in second_step["fallback_reason"].lower()
    assert second_step["original_attempted_action_name"] == "q_exploratory"
    assert second_step["questions_used"] == 1


def test_interactive_mode_uses_fallback_recommendation_pending_action():
    start = client.post(
        "/api/demo/session/start",
        json={
            "policy": "always_ask",
            "user_profile": "balanced_viewer",
            "question_budget": 1,
            "seed": 123,
            "mode": "interactive",
        },
    )
    assert start.status_code == 200
    session_id = start.json()["session_id"]

    pending_first = client.post("/api/demo/session/next", json={"session_id": session_id})
    assert pending_first.status_code == 200
    apply_first = client.post(
        "/api/demo/session/respond",
        json={
            "session_id": session_id,
            "continuation": "continue",
            "question_feedback": "helpful",
            "hinted_genre": "scifi",
        },
    )
    assert apply_first.status_code == 200

    pending_second = client.post("/api/demo/session/next", json={"session_id": session_id})
    assert pending_second.status_code == 200
    pending_action = pending_second.json()["pending_action"]
    assert pending_action["action_type"] == "recommend"
    assert pending_action["fallback_applied"] is True
    assert pending_action["original_attempted_action_name"] == "q_exploratory"
    apply_second = client.post(
        "/api/demo/session/respond",
        json={
            "session_id": session_id,
            "continuation": "continue",
            "recommendation_feedback": "skipped",
        },
    )
    assert apply_second.status_code == 200

    step = apply_second.json()["latest_step"]
    assert step["action_type"] == "recommend"
    assert step["question_asked"] is False
    assert step["over_budget"] is False
    assert step["fallback_applied"] is True
    assert step["original_attempted_action_name"] == "q_exploratory"
    assert step["questions_used"] == 1

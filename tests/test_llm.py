import pytest
from pipeline.llm import MAX_HISTORY_TURNS


def test_history_turn_calculation():
    # 10 turns = 20 messages
    expected_messages = MAX_HISTORY_TURNS * 2
    assert expected_messages == 20

    history = []
    for i in range(15):
        history.append({"role": "user", "content": f"Q{i}"})
        history.append({"role": "assistant", "content": f"A{i}"})

    assert len(history) == 30
    trimmed = history[-(MAX_HISTORY_TURNS * 2):]
    assert len(trimmed) == 20
    assert trimmed[0]["content"] == "Q5"
    assert trimmed[-1]["content"] == "A14"

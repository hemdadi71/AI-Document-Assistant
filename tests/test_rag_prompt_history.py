from rag.prompt import build_rag_prompt


def test_rag_prompt_uses_history_only_for_followup_context():
    contexts = [{
        "document": {
            "source": "manual.pdf",
            "chunk_id": 3,
            "text": "The device uses a 12V power supply.",
        }
    }]
    history = [
        {"role": "user", "content": "Tell me about the device."},
        {"role": "assistant", "content": "Which device?"},
    ]

    prompt = build_rag_prompt(
        "What voltage does it need?",
        contexts,
        history,
    )

    assert "Tell me about the device." in prompt
    assert "Which device?" in prompt
    assert "Previous assistant messages are not evidence" in prompt
    assert "What voltage does it need?" in prompt
    assert "12V power supply" in prompt

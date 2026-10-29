"""Ageing a voice description up or down (voice_age.shift_age)."""
import pytest

from androidreader_server.voice_age import current_age, shift_age


@pytest.mark.parametrize("text,step,expected", [
    ("An adult female voice. Way of speaking: brief.", -1, "A young adult female voice. Way of speaking: brief."),
    ("An adult female voice.", 1, "A middle-aged female voice."),
    ("A young man in his twenties with a bright voice.", -1, "A teenage boy in his teens with a bright voice."),
    ("A young man in his twenties with a bright voice.", 1, "An adult man in his thirties with a bright voice."),
    ("An elderly man with a low, rough voice.", -1, "A middle-aged man with a low, rough voice."),
    ("A child of about ten with a high, clear voice.", 1, "A teenager of about 15 with a high, clear voice."),
    ("A bright, energetic child's voice.", 1, "A bright, energetic teenager's voice."),
    ("A 16-year-old girl, sarcastic.", 1, "A 24-year-old young woman, sarcastic."),
    ("male, middle-aged, low pitch, american accent", 1, "male, elderly, low pitch, american accent"),
    # the age the description is about is its first clear one; the grandmother is a manner
    ("A young girl's high voice, old-fashioned like a grandmother", 1,
     "A teenage girl's high voice, old-fashioned like a grandmother"),
    ("A teenage girl, bright", -1, "A little girl, bright"),
])
def test_shift_age(text, step, expected):
    assert shift_age(text, step)["description"] == expected


def test_age_limits_and_no_age():
    with pytest.raises(ValueError):
        shift_age("A little boy", -1)
    with pytest.raises(ValueError):
        shift_age("An old woman, about seventy years old", 1)
    r = shift_age("A warm voice with a british accent", -1)
    assert r["from"] is None and r["description"].endswith("The voice sounds noticeably younger.")
    assert current_age("A smooth, youthful male voice") == "young adult"
    assert current_age("An old-fashioned, formal voice") is None

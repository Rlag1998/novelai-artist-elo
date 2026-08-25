"""
Fail-first tests for the "None" auto-negative preset.

The UI offers a "None" option (index -1) that is meant to send no NovelAI
preset, so only the user's own negative prompt applies. Before this fix the
ranker mapped -1 to ``ucPreset=None``, which novelai-python treats as
"unspecified" and silently replaces with preset 0 (Heavy). These tests fail
against that code.
"""
import asyncio

import pytest

import config
import artist_elo_ranker as ranker

REAL_PRESETS = (0, 1, 2, 3)
FAKE_PNG = b"\x89PNG\r\n\x1a\nfake-bytes"


class _FakeResp:
    def __init__(self, data: bytes):
        self.files = [("image_0.png", data)]


def _capture_request(monkeypatch, store: dict):
    async def fake_request(self, session=None, **kwargs):
        store["payload"] = self.model_dump(mode="json", exclude_none=True)
        return _FakeResp(FAKE_PNG)

    monkeypatch.setattr(ranker.GenerateImageInfer, "request", fake_request)


def _params(gen):
    return gen.model_dump(mode="json", exclude_none=True)["parameters"]


def test_preset_none_sends_only_the_users_negative_prompt():
    params = _params(ranker.build_generation(prompt="1girl", negative_prompt="lowres, bad hands", uc_preset=-1, seed=1))
    assert params["negative_prompt"] == "lowres, bad hands"
    assert params["v4_negative_prompt"]["caption"]["base_caption"] == "lowres, bad hands"
    assert params.get("ucPreset") not in REAL_PRESETS, "'None' must not be sent as a real preset"


def test_preset_none_with_default_negative_prompt_adds_no_preset_text():
    params = _params(ranker.build_generation(prompt="1girl", negative_prompt=None, uc_preset=-1, seed=1))
    assert params["negative_prompt"] == config.NEGATIVE_PROMPT
    assert params["v4_negative_prompt"]["caption"]["base_caption"] == config.NEGATIVE_PROMPT


def test_heavy_preset_still_merges_preset_text():
    """Guard: the fix must not disturb the real presets."""
    params = _params(ranker.build_generation(prompt="1girl", negative_prompt="lowres, bad hands", uc_preset=0, seed=1))
    assert params["ucPreset"] == 0
    assert params["negative_prompt"] != "lowres, bad hands"
    assert params["negative_prompt"].endswith("lowres, bad hands")


def test_generate_image_with_preset_none_reaches_the_request(tmp_path, monkeypatch):
    """The UI path: generate_image(uc_preset=-1) must send the user's negative untouched."""
    store = {}
    _capture_request(monkeypatch, store)
    ok = asyncio.run(
        ranker.generate_image(session=None, prompt="1girl", output_path=tmp_path / "x.png",
                              negative_prompt="lowres", quality_toggle=True, uc_preset=-1)
    )
    assert ok is True
    params = store["payload"]["parameters"]
    assert params["negative_prompt"] == "lowres"
    assert params["v4_negative_prompt"]["caption"]["base_caption"] == "lowres"
    assert params.get("ucPreset") not in REAL_PRESETS

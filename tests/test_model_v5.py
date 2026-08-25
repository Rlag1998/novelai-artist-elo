"""
Fail-first tests for the NovelAI Diffusion V5 Full migration.

These were written before the implementation. Against the pre-migration code
they fail because the ranker still sends ``nai-diffusion-4-5-full`` and has no
``MODEL_ID`` / ``build_generation`` surface.

No test here touches the network except ``test_live_generation_is_v5``, which
is opt-in (``NAI_LIVE_TEST=1`` plus a real ``NOVELAI_API_KEY``).
"""
import asyncio
import importlib
import os

import pytest
from novelai_python import GenerateImageInfer
from novelai_python.sdk.ai.generate_image import Model, Sampler, UCPreset

import config
import artist_elo_ranker as ranker

V5_FULL = "nai-diffusion-5-full"
V45_FULL = "nai-diffusion-4-5-full"
FAKE_PNG = b"\x89PNG\r\n\x1a\nfake-bytes"


class _FakeResp:
    """Minimal stand-in for ImageGenerateResp: only ``.files`` is read."""

    def __init__(self, data: bytes):
        self.files = [("image_0.png", data)]


def _capture_request(monkeypatch, store: dict):
    """Replace the network call with a recorder that returns fake PNG bytes."""

    async def fake_request(self, session=None, **kwargs):
        store["payload"] = self.model_dump(mode="json", exclude_none=True)
        return _FakeResp(FAKE_PNG)

    monkeypatch.setattr(ranker.GenerateImageInfer, "request", fake_request)


# ------------------------------------------------------------------------------
# Fail-first: these fail on the pre-migration code
# ------------------------------------------------------------------------------

def test_default_model_id_is_v5_full():
    assert config.MODEL_ID == V5_FULL
    assert ranker.MODEL_ID == V5_FULL


def test_generate_image_sends_v5_full_with_v4_family_payload(tmp_path, monkeypatch):
    """The request that leaves generate_image() must name V5 Full and keep the
    V4-family fields (v4_prompt, v4_negative_prompt) that V5 expects."""
    store = {}
    _capture_request(monkeypatch, store)
    out = tmp_path / "out.png"

    ok = asyncio.run(
        ranker.generate_image(
            session=None,
            prompt="1girl, artist:foo, test",
            output_path=out,
            negative_prompt="lowres",
            quality_toggle=True,
            uc_preset=0,
        )
    )

    assert ok is True
    assert out.read_bytes() == FAKE_PNG

    payload = store["payload"]
    assert payload["model"] == V5_FULL

    params = payload["parameters"]
    assert params["params_version"] == 3
    assert "v4_prompt" in params, (
        "v4_prompt missing: novelai-python fell back to the pre-V4 payload, "
        "which V5 rejects"
    )
    assert params["v4_prompt"]["caption"]["base_caption"].startswith("1girl, artist:foo, test")
    assert "v4_negative_prompt" in params
    assert params["v4_negative_prompt"]["caption"]["base_caption"].endswith("lowres")


def test_build_generation_reuses_v45_payload_except_model():
    """The V5 request must be the V4.5 request with only the model id changed."""
    kwargs = dict(
        prompt="1girl, artist:foo",
        negative_prompt="lowres",
        quality_toggle=True,
        uc_preset=0,
        seed=1234,
    )
    v5 = ranker.build_generation(**kwargs).model_dump(mode="json", exclude_none=True)
    baseline = ranker.build_generation(model_id=V45_FULL, **kwargs).model_dump(
        mode="json", exclude_none=True
    )

    assert v5.pop("model") == V5_FULL
    assert baseline.pop("model") == V45_FULL
    assert v5 == baseline


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_build_generation_preserves_uc_preset_index(index):
    """The UI's preset index must reach the request unchanged.

    Index -1 ("None") is covered in test_uc_preset_none.py."""
    gen = ranker.build_generation(prompt="1girl", negative_prompt="lowres", uc_preset=index, seed=1)
    params = gen.model_dump(mode="json", exclude_none=True)["parameters"]
    assert params["ucPreset"] == index
    assert params["negative_prompt"].endswith("lowres")


def test_model_id_env_override(monkeypatch):
    """NAI_MODEL lets a user pin a different model without editing code."""
    monkeypatch.setenv("NAI_MODEL", V45_FULL)
    try:
        cfg = importlib.reload(config)
        assert cfg.MODEL_ID == V45_FULL
    finally:
        monkeypatch.delenv("NAI_MODEL", raising=False)
        importlib.reload(config)
    assert config.MODEL_ID == V5_FULL


# ------------------------------------------------------------------------------
# Library characterisation: documents why the workaround exists
# ------------------------------------------------------------------------------

def test_raw_v5_string_in_build_generate_drops_v4_fields():
    """novelai-python <= 0.7.12 does not know the V5 id. Passing it straight to
    build_generate() silently produces a pre-V4 payload. When this test starts
    to fail, the installed library has learned V5 and the swap in
    build_generation() can be retired."""
    if V5_FULL in {m.value for m in Model}:
        pytest.skip("installed novelai-python knows V5; workaround is obsolete")

    gen = GenerateImageInfer.build_generate(
        prompt="1girl",
        model=V5_FULL,
        width=1024,
        height=1024,
        steps=28,
        sampler=Sampler.K_EULER_ANCESTRAL,
        negative_prompt="lowres",
        ucPreset=UCPreset.TYPE0,
        qualityToggle=True,
    )
    params = gen.model_dump(mode="json", exclude_none=True)["parameters"]
    assert "v4_prompt" not in params


# ------------------------------------------------------------------------------
# Live check (opt-in): proves the API actually renders with V5
# ------------------------------------------------------------------------------

@pytest.mark.skipif(
    os.getenv("NAI_LIVE_TEST") != "1" or not os.getenv("NOVELAI_API_KEY"),
    reason="live NovelAI call; set NAI_LIVE_TEST=1 and NOVELAI_API_KEY to run",
)
def test_live_generation_is_v5(tmp_path):
    from PIL import Image
    from pydantic import SecretStr
    from novelai_python import ApiCredential

    session = ApiCredential(api_token=SecretStr(config.get_api_key()))
    out = tmp_path / "live.png"

    ok = asyncio.run(
        ranker.generate_image(
            session=session,
            prompt="1girl, solo, portrait, simple background",
            output_path=out,
        )
    )

    assert ok is True
    source = Image.open(out).info.get("Source", "")
    assert source.startswith("NovelAI Diffusion V5"), source

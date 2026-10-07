"""The contract the AEDSmartX Node backend relies on: bearer auth, the
multi-file `file` field (one photo, one readiness video, or a frame
sequence), the `aedModel` form field, and the profile echoed back."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services import gemini_checklist_service as svc
from app.services.device_profiles import get_profile

TOKEN = "test-token"


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setattr(settings, "CV_SERVICE_TOKEN", TOKEN)


@pytest.fixture
def client():
    return TestClient(app)


def _auth():
    return {"Authorization": f"Bearer {TOKEN}"}


def _verdict(**fields):
    return svc.ChecklistAnalysisResult(passed=True, confidence=0.9, notes="ok", **fields)


def test_rejects_missing_token(client):
    res = client.post("/api/v1/checklist/serial_number/analyze", files={"file": ("a.jpg", b"x", "image/jpeg")})
    assert res.status_code == 401


def test_photo_passes_profile_and_echoes_it(client):
    with patch.object(svc, "analyze_checklist_item", AsyncMock(return_value=_verdict(serial_number="B17C-00516"))) as m:
        res = client.post(
            "/api/v1/checklist/serial_number/analyze",
            headers=_auth(),
            files={"file": ("a.jpg", b"jpeg", "image/jpeg")},
            data={"aedModel": "Philips FRx"},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["serial_number"] == "B17C-00516"
    assert body["profile"] == {"id": "Philips FRx", "name": "Philips HeartStart FRx"}
    assert m.call_args.kwargs["aed_model"] == "Philips FRx"
    assert m.call_args.kwargs.get("frames") is None


def test_frame_sequence_goes_to_frames_path(client):
    files = [("file", (f"f{i}.jpg", b"jpeg", "image/jpeg")) for i in range(5)]
    with patch.object(svc, "analyze_checklist_item", AsyncMock(return_value=_verdict(status="ready"))) as m:
        res = client.post(
            "/api/v1/checklist/readiness_indicator/analyze",
            headers=_auth(),
            files=files,
            data={"aedModel": "Zoll AED Plus"},
        )
    assert res.status_code == 200
    assert len(m.call_args.kwargs["frames"]) == 5


def test_single_video_goes_to_video_path(client):
    with patch.object(svc, "analyze_checklist_item", AsyncMock(return_value=_verdict(status="ready"))) as m:
        res = client.post(
            "/api/v1/checklist/readiness_indicator/analyze",
            headers=_auth(),
            files={"file": ("clip.mp4", b"mp4", "video/mp4")},
        )
    assert res.status_code == 200
    args = m.call_args
    assert args.args[1] == b"mp4" and args.args[2] == "video/mp4"
    assert args.kwargs.get("frames") is None


def test_video_refused_for_photo_item(client):
    res = client.post(
        "/api/v1/checklist/pads_expiry/analyze",
        headers=_auth(),
        files={"file": ("clip.mp4", b"mp4", "video/mp4")},
    )
    assert res.status_code == 400


def test_daily_limit_is_503(client):
    with patch.object(svc, "analyze_checklist_item", AsyncMock(side_effect=svc.DailyLimitReached("x"))):
        res = client.post(
            "/api/v1/checklist/serial_number/analyze",
            headers=_auth(),
            files={"file": ("a.jpg", b"jpeg", "image/jpeg")},
        )
    assert res.status_code == 503


@pytest.mark.parametrize(
    "key,expected",
    [
        ("frx", "Philips FRx"),
        ("hs1", "Philips HS1"),
        ("zollPlus", "Zoll AED Plus"),
        ("g5", "Zoll Powerheart G5"),
        ("defibtech", "Defibtech Lifeline"),
        ("crPlus", "generic"),
        ("Defibtech Lifeline AUTO", "Defibtech Lifeline AUTO"),
    ],
)
def test_legacy_client_keys_map_to_profiles(key, expected):
    assert get_profile(key).id == expected


async def test_frames_are_numbered_and_never_flash_scanned():
    response = MagicMock()
    response.parsed = svc.ChecklistVerdict(
        passed=True, confidence=0.9, notes="Green check visible.", status="ready", ready_frames=[2]
    )
    response.usage_metadata = None
    with patch.object(svc, "_get_client") as get_client, patch.object(svc.readiness_frames, "prepare") as prepare:
        client = AsyncMock()
        client.aio.models.generate_content = AsyncMock(return_value=response)
        get_client.return_value = client
        result = await svc.analyze_checklist_item(
            "readiness_indicator", b"", None, aed_model="Zoll AED Plus", frames=[b"a", b"b", b"c"]
        )
    prepare.assert_not_called()
    contents = client.aio.models.generate_content.call_args.kwargs["contents"]
    labels = [p.text for p in contents if getattr(p, "text", None) and p.text.startswith("Frame")]
    assert labels == ["Frame 1", "Frame 2", "Frame 3"]
    assert result.status == "ready" and result.passed
    assert result.meta.frames_sent == 3


async def test_frames_ready_without_cited_frame_is_unclear():
    response = MagicMock()
    response.parsed = svc.ChecklistVerdict(passed=True, confidence=0.9, notes="Looks ready.", status="ready")
    response.usage_metadata = None
    with patch.object(svc, "_get_client") as get_client:
        client = AsyncMock()
        client.aio.models.generate_content = AsyncMock(return_value=response)
        get_client.return_value = client
        result = await svc.analyze_checklist_item(
            "readiness_indicator", b"", None, aed_model="Philips FRx", frames=[b"a", b"b"]
        )
    assert result.status == "unclear" and not result.passed

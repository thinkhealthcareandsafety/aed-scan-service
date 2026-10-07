"""The readiness check must not pass a unit whose ready signal was never seen.

Found on 30 Sep 2026: twenty frames spread evenly over a 10-second clip of a
blinking Philips HS1 contained not one flash, and the prompt told the model
gaps were normal — so a healthy unit failed, and a unit whose light never
came on could pass. These tests hold the fix in place: every frame is
scanned for flashes, a "ready" must name the frames that show it, and a
unit that proves readiness by blinking must actually have blinked.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.services import gemini_checklist_service as svc
from app.services import readiness_frames
from app.services.checklist_items import get_item
from app.services.device_profiles import PHILIPS_FRX, PHILIPS_HS1, ZOLL_AED_PLUS, get_profile

ITEM = get_item("readiness_indicator")


def _verdict(status="ready", passed=True, ready_frames=None, notes="Ready light seen."):
    return svc.ChecklistAnalysisResult(
        passed=passed, confidence=0.9, notes=notes, status=status, ready_frames=ready_frames
    )


def _video(flashes: int, frames: int = 24) -> readiness_frames.ReadinessFrames:
    return readiness_frames.ReadinessFrames(
        frames=[b"x"] * frames, times=[i * 0.4 for i in range(frames)], flash_count=flashes
    )


def _check(result, profile, video, language=None):
    return svc._apply_deterministic_checks(ITEM, result, profile=profile, video=video, language=language)


# ── A pass must be backed by evidence ───────────────────────────────────────

@pytest.mark.unit
def test_a_ready_verdict_that_names_no_frame_is_not_a_pass():
    checked = _check(_verdict(ready_frames=None), PHILIPS_HS1, _video(flashes=2))
    assert checked.passed is False
    assert checked.status == "unclear"
    assert "Ready light" in checked.notes


@pytest.mark.unit
@pytest.mark.parametrize("profile", [PHILIPS_FRX, PHILIPS_HS1])
def test_a_philips_that_never_blinked_cannot_pass_whatever_the_model_says(profile):
    # The model claims it saw the light in frame 5 — the green On/Off button,
    # a reflection — but no frame of the video had a flash in it.
    checked = _check(_verdict(ready_frames=[5]), profile, _video(flashes=0))
    assert checked.passed is False
    assert checked.status == "unclear"
    assert "flash" in checked.notes


@pytest.mark.unit
def test_a_philips_that_blinked_and_was_seen_passes():
    checked = _check(_verdict(ready_frames=[4, 11]), PHILIPS_HS1, _video(flashes=3))
    assert checked.passed is True
    assert checked.status == "ready"


@pytest.mark.unit
def test_a_zoll_is_judged_on_its_steady_check_not_on_blinking():
    # The AED Plus never blinks: a green check seen in the window is a pass
    # even though no flash was found.
    checked = _check(_verdict(ready_frames=[2, 9]), ZOLL_AED_PLUS, _video(flashes=0))
    assert checked.passed is True
    assert checked.status == "ready"


@pytest.mark.unit
def test_a_zoll_ready_with_no_frame_asks_for_the_status_window():
    checked = _check(_verdict(ready_frames=[]), ZOLL_AED_PLUS, _video(flashes=0))
    assert checked.passed is False
    assert "status window" in checked.notes


@pytest.mark.unit
def test_frame_numbers_that_do_not_exist_are_not_evidence():
    checked = _check(_verdict(ready_frames=[31, 40]), PHILIPS_HS1, _video(flashes=2, frames=24))
    assert checked.passed is False


@pytest.mark.unit
def test_passed_follows_status_not_the_other_way_round():
    checked = _check(_verdict(status="fault", passed=True, ready_frames=[3]), PHILIPS_HS1, _video(flashes=2))
    assert checked.passed is False
    assert checked.status == "fault"


@pytest.mark.unit
def test_a_pass_with_no_status_and_no_evidence_is_unclear():
    checked = _check(_verdict(status=None, passed=True, ready_frames=None), get_profile(None), None)
    assert checked.passed is False
    assert checked.status == "unclear"


@pytest.mark.unit
def test_a_fault_is_never_turned_into_a_pass():
    checked = _check(_verdict(status="fault", passed=False, ready_frames=None), PHILIPS_HS1, _video(flashes=3))
    assert checked.passed is False
    assert checked.status == "fault"


@pytest.mark.unit
def test_the_retake_advice_is_in_hindi_when_the_inspector_reads_hindi():
    checked = _check(_verdict(ready_frames=None), PHILIPS_FRX, _video(flashes=0), language="hi")
    assert checked.notes_hi and "Ready लाइट" in checked.notes_hi


@pytest.mark.unit
def test_the_prompt_no_longer_tells_the_model_gaps_are_fine():
    prompt = svc._build_prompt(ITEM, PHILIPS_HS1, frame_count=24, duration=10)
    assert "ready_frames" in prompt
    assert "Never conclude that a light blinked between frames" in prompt
    assert "gaps between flashes are normal" not in prompt


# ── The scan finds real flashes, and only real flashes ─────────────────────

def _clip(tmp_path, name, flash_at=(), green_button=True, drift=1.0, seconds=5, fps=30, smooth=False):
    """A small generated clip of a device front: a blue panel with seams and
    a label, a steady green On/Off button, and a Ready light that is dark
    except for 3-frame flashes (with the glow a real LED has), filmed by a
    hand that drifts. `smooth` leaves out the seams and label — a bland
    surface is where steadying the hand is hardest."""
    rng = np.random.default_rng(3)
    base = np.full((360, 480, 3), (120, 60, 30), np.uint8)  # BGR: device blue
    if not smooth:
        cv2.rectangle(base, (40, 40), (220, 320), (200, 200, 200), -1)  # label
        for y in range(70, 300, 28):
            cv2.line(base, (60, y), (200, y), (40, 40, 40), 2)  # printed text
        cv2.line(base, (250, 0), (250, 360), (70, 35, 15), 3)  # seam
        cv2.rectangle(base, (270, 250), (440, 330), (90, 45, 25), 2)
    if green_button:
        cv2.circle(base, (330, 200), 30, (40, 150, 40), -1)
    cv2.ellipse(base, (330, 140), (9, 4), 0, 0, 360, (30, 40, 30), -1)  # the light's dark window
    lit = {int(t * fps) + k for t in flash_at for k in range(3)}
    n = seconds * fps
    path = str(tmp_path / name)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (400, 300))
    pos = np.cumsum(rng.normal(0, drift, (n, 2)), axis=0)
    for i in range(n):
        img = base.copy()
        if i in lit:
            glow = np.zeros_like(img)
            cv2.ellipse(glow, (330, 140), (14, 8), 0, 0, 360, (60, 255, 90), -1)
            img = cv2.add(img, cv2.GaussianBlur(glow, (0, 0), 4))
            cv2.ellipse(img, (330, 140), (8, 3), 0, 0, 360, (190, 255, 200), -1)
        dx, dy = pos[i]
        m = np.float32([[1, 0, -40 + dx], [0, 1, -30 + dy]])
        frame = cv2.warpAffine(img, m, (400, 300), borderMode=cv2.BORDER_REFLECT)
        noise = rng.normal(0, 3, frame.shape).astype(np.int16)
        out.write(np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8))
    out.release()
    with open(path, "rb") as f:
        return f.read()


@pytest.mark.unit
def test_every_flash_of_a_blinking_light_is_found(tmp_path):
    video = readiness_frames.prepare(_clip(tmp_path, "blink.mp4", flash_at=(1.0, 3.0)))
    assert video is not None
    assert video.flash_count == 2
    flash_times = [video.times[p - 1] for p in video.flash_positions]
    assert any(abs(t - 1.03) < 0.1 for t in flash_times)
    assert any(abs(t - 3.03) < 0.1 for t in flash_times)


@pytest.mark.unit
def test_a_light_that_never_comes_on_has_no_flashes(tmp_path):
    video = readiness_frames.prepare(_clip(tmp_path, "dark.mp4"))
    assert video is not None
    assert video.flash_count == 0


@pytest.mark.unit
def test_a_steady_green_button_under_a_shaking_hand_is_not_a_flash(tmp_path):
    video = readiness_frames.prepare(_clip(tmp_path, "shaky.mp4", drift=3.0))
    assert video is not None
    assert video.flash_count == 0


@pytest.mark.unit
def test_a_bland_surface_that_is_hard_to_steady_raises_no_false_flash(tmp_path):
    video = readiness_frames.prepare(_clip(tmp_path, "smooth.mp4", drift=2.0, smooth=True))
    assert video is not None
    assert video.flash_count == 0


@pytest.mark.unit
def test_an_undecodable_video_falls_back_to_the_raw_video():
    assert readiness_frames.prepare(b"not a video") is None

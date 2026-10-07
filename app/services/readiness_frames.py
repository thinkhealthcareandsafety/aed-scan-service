"""
Frames for the readiness-indicator check — chosen so a blink can't be lost.

A Philips Ready light flashes for about a tenth of a second, every few
seconds. Twenty frames spread evenly over a 10-second clip land half a
second apart, so the flash usually falls between them: on a test clip of a
blinking HS1, not one of the twenty showed the light lit. The model then
saw a healthy unit and a dead one as the same dark light — and the prompt,
loosened to stop it failing healthy units, told it gaps were normal, which
is how a unit that never blinked could pass.

So every frame is scanned. A flash is a small spot that becomes brighter
AND greener than anything that is usually around it, then goes back. The
scan steadies handheld shake (phase correlation) and compares each frame
with the clip's own median, dilated: a steady green thing that merely
shifts a pixel — the green On/Off button on an FRx, the PULL handle on an
HS1 — never exceeds its own neighbourhood, but a light switching on in a
dark window does. The frames where that happens go to the model with
evenly spaced ones, and are kept as evidence the verdict is checked against.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

#: Frames sent to the model: flash moments first, the rest spread evenly.
MAX_FRAMES = 24
MAX_FLASH_FRAMES = 6
#: Long edge of the frames the model sees. A status light is unmistakable
#: at this size, and the request stays small on a field connection.
SEND_LONG_EDGE = 640
#: Long edge the scan works at.
SCAN_LONG_EDGE = 256
#: Frames scanned at most; a longer clip is scanned every n-th frame, which
#: still leaves ~30 a second — more than enough to catch a flash.
MAX_SCANNED = 900
#: Frames the clip's usual appearance (its median) is built from.
BACKGROUND_SAMPLES = 60
#: How far around each pixel counts as "usually around it", in scan pixels.
NEIGHBOURHOOD = 5

#: Lit = this much greener (G minus the larger of R and B) and this much
#: brighter than the brightest, greenest thing usually nearby.
GREENER_BY = 20
BRIGHTER_BY = 15
#: And a lit Ready light is vivid in itself — strongly green and bright,
#: not just a shade greener than before — which dull plastic or a patch of
#: wall never is.
MIN_GREEN = 45
MIN_BRIGHT = 140
#: A light, not a scene change: at least this many lit pixels at scan size,
#: no more than this share of the frame, and gathered in one small spot.
MIN_LIT_PIXELS = 2
MAX_LIT_SHARE = 0.012
MAX_SPOT_SIDE = 0.1  # of the frame's long edge
#: Below this phase-correlation response the shake correction isn't
#: trustworthy, and neither is anything that seems to change in the frame.
MIN_ALIGNMENT = 0.08
#: A frame where this share of pixels changed brightness is a camera jolt or
#: an exposure swing: nothing in it is trusted as a flash.
UNSTABLE_SHARE = 0.3
#: A flash must be dark again this long before and after it is seen lit.
OFF_AFTER_SECONDS = 0.35
#: Lit longer than this is a steady light (in use, self-test), not a blink.
MAX_FLASH_SECONDS = 0.7
#: A Ready flash lasts about a tenth of a second, so it spans at least two
#: frames; a spot lit in one frame alone is a jolt of the hand.
MIN_FLASH_FRAMES = 2


@dataclass
class ReadinessFrames:
    #: JPEGs for the model, in time order.
    frames: List[bytes]
    #: Seconds into the clip of each frame above.
    times: List[float]
    #: 1-based positions (in `frames`) of frames showing a detected flash.
    flash_positions: List[int] = field(default_factory=list)
    #: Separate flashes found in the whole clip.
    flash_count: int = 0
    duration: float = 0.0


def _resize(frame: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = long_edge / max(h, w)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def _green_and_bright(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b, g, r = cv2.split(bgr.astype(np.int16))
    green = np.clip(g - np.maximum(r, b), 0, 255).astype(np.uint8)
    bright = bgr.max(axis=2)
    return green, bright


def _lit(green: np.ndarray, bright: np.ndarray, near_green: np.ndarray, near_bright: np.ndarray) -> np.ndarray:
    """Pixels vividly green and bright, and more so than anything usually
    around them."""
    return (
        (green >= MIN_GREEN)
        & (bright >= MIN_BRIGHT)
        & ((green.astype(np.int16) - near_green) > GREENER_BY)
        & ((bright.astype(np.int16) - near_bright) > BRIGHTER_BY)
    )


def _one_small_spot(lit: np.ndarray) -> bool:
    """Most lit pixels in one blob, small enough to be a light."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(lit.astype(np.uint8), connectivity=8)
    if count < 2:
        return False
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[biggest, cv2.CC_STAT_AREA])
    side = max(int(stats[biggest, cv2.CC_STAT_WIDTH]), int(stats[biggest, cv2.CC_STAT_HEIGHT]))
    return (
        area >= MIN_LIT_PIXELS
        and area >= 0.6 * int(np.count_nonzero(lit))
        and side <= max(6, MAX_SPOT_SIDE * max(lit.shape))
    )


def _find_flashes(green: np.ndarray, bright: np.ndarray, off_gap: int, steady: Optional[List[bool]] = None) -> List[int]:
    """Positions (in the scanned stack) of frames with a flash.

    Two tests. Against the clip's usual picture: a spot is lit that normally
    isn't. Against the same spot `off_gap` frames before and after: it was
    dark then — a blink goes on and off again. Hand drift can bring a green
    button or handle into a place it wasn't, but it stays there, so it fails
    the second test; a flash passes both."""
    count = green.shape[0]
    picks = np.linspace(0, count - 1, min(BACKGROUND_SAMPLES, count)).astype(int)
    bg_green = np.median(green[picks], axis=0).astype(np.uint8)
    bg_bright = np.median(bright[picks], axis=0).astype(np.uint8)
    kernel = np.ones((NEIGHBOURHOOD, NEIGHBOURHOOD), np.uint8)
    near_green = cv2.dilate(bg_green, kernel).astype(np.int16)
    near_bright = cv2.dilate(bg_bright, kernel).astype(np.int16)
    area = bg_green.size

    found: List[int] = []
    # Only frames with a "before" and an "after" to check against: a spot
    # that can't be seen going dark again is not proven to be a blink.
    for k in range(off_gap, count - off_gap):
        if steady is not None and not (steady[k] and steady[k - off_gap] and steady[k + off_gap]):
            continue
        changed = np.abs(bright[k].astype(np.int16) - bg_bright.astype(np.int16)) > 25
        if np.count_nonzero(changed) > UNSTABLE_SHARE * area:
            continue
        lit = _lit(green[k], bright[k], near_green, near_bright)
        if not MIN_LIT_PIXELS <= int(np.count_nonzero(lit)) <= MAX_LIT_SHARE * area:
            continue
        # On, then off: the spot must be dark just before and just after.
        for j in (k - off_gap, k + off_gap):
            lit &= _lit(green[k], bright[k], cv2.dilate(green[j], kernel).astype(np.int16),
                        cv2.dilate(bright[j], kernel).astype(np.int16))
        if _one_small_spot(lit):
            found.append(k)
    return found


def _events(positions: List[int], gap: int) -> List[List[int]]:
    """Groups consecutive flash frames into separate flashes."""
    events: List[List[int]] = []
    for p in positions:
        if events and p - events[-1][-1] <= gap:
            events[-1].append(p)
        else:
            events.append([p])
    return events


def prepare(video_bytes: bytes) -> Optional[ReadinessFrames]:
    """Decode a readiness clip in one pass: find its flashes and pick the
    frames the model sees. None if it can't be decoded (the caller then
    sends the video itself)."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if total <= 0:
            cap.release()
            return None
        stride = max(1, -(-total // MAX_SCANNED))

        prev_gray = window = None
        size = None
        # Where the hand has moved the scene since the first frame, built up
        # frame by frame: matched against the first frame directly, the
        # match fades as the hand drifts (measured: near zero by 8 s), and
        # late frames went unsteadied.
        total_dx = total_dy = 0.0
        greens: List[np.ndarray] = []
        aligned_ok: List[bool] = []
        brights: List[np.ndarray] = []
        jpegs: List[bytes] = []
        indices: List[int] = []
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                ok_jpg, buf = cv2.imencode(".jpg", _resize(frame, SEND_LONG_EDGE), [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok_jpg:
                    small = _resize(frame, SCAN_LONG_EDGE)
                    if size is None:
                        size = (small.shape[1], small.shape[0])
                        window = cv2.createHanningWindow(size, cv2.CV_32F)
                    elif (small.shape[1], small.shape[0]) != size:
                        small = cv2.resize(small, size)
                    # Steady the hand: shift each frame back onto the first,
                    # by how far it moved since the frame before.
                    gray = np.float32(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
                    if prev_gray is None:
                        aligned_ok.append(True)
                    else:
                        (dx, dy), response = cv2.phaseCorrelate(prev_gray, gray, window)
                        total_dx += dx
                        total_dy += dy
                        aligned_ok.append(response >= MIN_ALIGNMENT)
                    prev_gray = gray
                    shift = np.float32([[1, 0, -total_dx], [0, 1, -total_dy]])
                    steady = cv2.blur(cv2.warpAffine(small, shift, size, borderMode=cv2.BORDER_REPLICATE), (3, 3))
                    green, bright = _green_and_bright(steady)
                    greens.append(green)
                    brights.append(bright)
                    jpegs.append(buf.tobytes())
                    indices.append(index)
            index += 1
        cap.release()
        if len(jpegs) < 2:
            return None

        scan_fps = fps / stride
        off_gap = max(2, round(OFF_AFTER_SECONDS * scan_fps))
        positions = (
            _find_flashes(np.stack(greens), np.stack(brights), off_gap, aligned_ok) if len(jpegs) >= 3 else []
        )
        # A blink is brief; anything lit for longer is a steady light or a
        # scene change, never a Ready flash.
        events = [
            e
            for e in _events(positions, gap=max(2, int(scan_fps * 0.25)))
            if len(e) >= MIN_FLASH_FRAMES and (e[-1] - e[0] + 1) / scan_fps <= MAX_FLASH_SECONDS
        ]

        # One frame per flash — the middle of its run — then even spacing.
        chosen = {e[len(e) // 2] for e in events[:MAX_FLASH_FRAMES]}
        flash_set = set(chosen)
        for p in np.linspace(0, len(jpegs) - 1, MAX_FRAMES).astype(int):
            if len(chosen) >= MAX_FRAMES:
                break
            if all(abs(int(p) - c) > 1 for c in chosen):
                chosen.add(int(p))

        ordered = sorted(chosen)
        result = ReadinessFrames(
            frames=[jpegs[p] for p in ordered],
            times=[round(indices[p] / fps, 2) for p in ordered],
            flash_positions=[i + 1 for i, p in enumerate(ordered) if p in flash_set],
            flash_count=len(events),
            duration=round(index / fps, 1),
        )
        logger.info("readiness.frames", scanned=len(jpegs), flashes=len(events), sent=len(result.frames))
        return result
    except Exception as exc:  # noqa: BLE001 — any decode failure falls back
        logger.warning("readiness.prepare_failed", error=str(exc))
        return None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

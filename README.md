# AED Scan Service (beta)

Internal microservice: one Gemini-driven analysis call per uploaded photo,
scoped to exactly 3 checklist items (`serial_number`, `pads_expiry`,
`battery_expiry`). Called only by the AEDSmartX Node backend
(`smartxb2c-server`) — never directly by a browser.

Adapted and trimmed from
[thinkhealthcareandsafety/Aed-inspection-platform](https://github.com/thinkhealthcareandsafety/Aed-inspection-platform)'s
`python-cv` service (10 items, LED/video detection, own auth/device model) down
to just the 3 items AEDSmartX's manual inspection screen already tracks, for
a single-account Phase 1 beta.

## IMPORTANT — do not expose this service publicly

This service has one layer of its own auth (a shared bearer token, see
`app/auth.py`), but that is a second line of defense, not the first. It
should run on a private network / internal-only Render service, reachable
only from the Node backend — never bound to a public port. An unauthenticated
request in front of an exposed port is a free way for anyone to burn your
Gemini quota.

## Local development

```bash
python -m venv .venv
./.venv/Scripts/activate   # Windows; source .venv/bin/activate on Mac/Linux
pip install -r requirements.txt

cp .env.example .env
# Fill in GEMINI_API_KEY (from https://aistudio.google.com) and
# CV_SERVICE_TOKEN (must exactly match CV_SERVICE_TOKEN in the Node
# backend's .env — that's what authenticates the backend to this service)

uvicorn app.main:app --port 8001
```

Health check: `GET /health` (no auth required).
Everything else requires `Authorization: Bearer <CV_SERVICE_TOKEN>`.

## Endpoint

`POST /api/v1/checklist/{item_id}/analyze` where `item_id` is one of
`serial_number`, `pads_expiry`, `battery_expiry`. Multipart form upload,
field name `file` (JPEG/PNG). Returns:

```json
{
  "passed": true,
  "confidence": 0.92,
  "notes": "Serial number clearly visible and legible.",
  "serial_number": "A25H-13331",
  "expiry_date": null,
  "expiry_raw_text": null,
  "lot_number": null,
  "battery_serial_number": null
}
```

## What was intentionally left out of Phase 1

- LED/status-indicator detection (video + frame extraction) — the least
  proven part of the source project; not needed for the 3 items this beta
  covers.
- Manufacturer plugin system, YOLO detection — the source project's earlier
  architecture already dropped these in favor of Gemini's vision API
  directly; nothing to carry over.
- Any device/organization/auth model — AEDSmartX's own `Product` /
  `Organization` records are the source of truth; this service only reads
  photos and returns what it saw.

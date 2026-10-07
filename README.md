# AED Scan Service

Internal microservice: one Gemini-driven analysis call per checklist item —
a label photo, or a readiness-light video. Called only by the AEDSmartX Node
backend (`smartxb2c-server`) — never directly by a browser.

## Engine

`app/services` and `app/utils` are kept file-for-file in step with
[thinkhealthcareandsafety/Aed-inspection-platform](https://github.com/thinkhealthcareandsafety/Aed-inspection-platform)'s
`python-cv` service, and its test suite runs here unchanged (`pytest`). What
that engine does, per check:

- **Device profiles** (`device_profiles.py`) — manufacturer-sourced guidance
  for the Philips FRx and HS1, ZOLL AED Plus and AED 3, Powerheart G3 and G5,
  and Defibtech Lifeline, Lifeline AUTO, Lifeline VIEW and Lifeline ECG: where
  each label is, which date counts, what "ready" looks like on that unit.
  Unknown models get a brand-neutral profile.
- **Dates read, never judged by the model** — ISO 15223-1 symbols separate
  manufacture from expiry; GS1 barcode fields `(17)`/`(15)`/`(11)` override a
  misread; the service decides expired / expiring soon against its own clock.
  Powerheart batteries (no printed expiry) are dated 4 years from install or
  manufacture.
- **Two-model reading** of serials and expiry dates: a second model reads the
  same label, and a disagreement is a retake, never a stored misread.
- **Readiness from video** — every frame is scanned (OpenCV) for the Ready
  light's flash; a blinking unit can't pass without one, and any "ready" must
  cite the frames that show it.
- **Wrong-unit detection** — a photo of a different maker's AED fails.
- **Audit metadata** on every verdict (`meta`): model, prompt version,
  latency, tokens, second-read outcome, and each rule that overruled the model.
- Hedged fallback across models, retry on Google-side errors, and a daily
  call ceiling (`DAILY_AI_CALL_LIMIT`).

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

`POST /api/v1/checklist/{item_id}/analyze` — any item in
`GET /api/v1/checklist/items`. Multipart form upload, field `file`: one photo;
or, for `readiness_indicator`, one video (preferred — scanned for flashes) or
up to 30 JPEG frames. Optional `aedModel`: a profile id from
`GET /api/v1/checklist/profiles` (e.g. `Philips FRx`). Returns the verdict plus
`meta` and the `profile` used, e.g.:

```json
{
  "passed": true,
  "confidence": 0.95,
  "notes": "Battery in date until 31 Dec 2028.",
  "expiry_date": "2028-12-31",
  "manufacture_date": "2023-03-01",
  "brand_seen": "Philips",
  "meta": {
    "model": "gemini-3.5-flash-lite",
    "prompt_version": "15185289eb",
    "latency_ms": 3016,
    "second_read": "agrees",
    "overrides": []
  },
  "profile": {"id": "Philips HS1", "name": "Philips HeartStart HS1 (OnSite)"}
}
```

Model choice and timing can be tuned without a deploy — see `.env.example`.

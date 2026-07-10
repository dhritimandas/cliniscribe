# Review Frontend — Plan and Shared Contracts (coordinator-owned)

Fully offline FastAPI + vanilla single-page app. "Quiet instrument" direction:
CAPTURE = Zen (centered mic ring, timer, three staged progress lines, nothing
else), REVIEW = Sage editorial (mono micro-labels, grid cells). The referenced
`cliniscribe_design_directions.html` is absent from the repo; the goal's textual
spec is the design authority.

## Hard constraints (every implementer obeys)
- Two accent colors ONLY: teal `#0F766E` (state), amber `#D97706` (attention).
  Everything else grayscale.
- Empty fields render "—", never red, never blank.
- Confidence is text + tint, never color alone. We have NO probability scores —
  only `low_confidence_fields` flags. Render flagged fields as amber tint +
  literal text "VERIFY"; unflagged get no badge. Do NOT invent percentages.
- Confidence/provenance detail appears on tap/click, not permanently.
- No dashboards, no counters, no charts. Progressive disclosure.
- Offline only: zero external URLs (no CDN fonts/scripts/styles). System font
  stack; monospace for micro-labels.

## Layout on disk
```
web/
  app.py            # FastAPI app (serves API + static)
  provenance.py     # note-field → transcript-turn matcher
  translations.py   # static label table en/hi/mr (dict, importable)
  static/
    index.html      # one page, two screens (capture, review)
    app.js          # vanilla JS, no build step
    styles.css
tests/test_web_app.py, tests/test_provenance.py
```

## Session state machine (server-authoritative)
`idle → recording → processing(stage) → review → signed`
State + stage live in `outputs/<session_id>/status.json`:
```json
{"state": "processing", "stage": "l3_asr", "stages_done": ["l1_preprocess","l2_diarize"], "error": null}
```
`pipeline.run` gains an optional `on_stage(name, event)` callback kwarg
(`event ∈ {"start","end"}`) — the ONLY pipeline change; default None keeps
current behavior. Backend uses it to update status.json.

## API (all JSON unless noted; no cloud calls anywhere)
| Route | Method | Body → Response |
|---|---|---|
| `/api/sessions` | POST | multipart `audio` (wav/mp3) → `{session_id}`; writes `outputs/<sid>/input.<ext>`, state=idle |
| `/api/sessions/{sid}/process` | POST | — → `202 {session_id}`; runs pipeline.run in a daemon thread with on_stage → status.json |
| `/api/sessions/{sid}/status` | GET | → status.json content |
| `/api/sessions/{sid}/note` | GET | → `{note: ClinicalNote-json, transcript: [turns], provenance: {field_path: {turn_index, start, end, snippet}}, flags: {field_path: reason, _general: [...]}, low_confidence_fields: [raw name-keyed list]}` — `flags` is the index-keyed dict from flags_by_path; the raw list rides along for fidelity |
| `/api/sessions/{sid}/note` | PATCH | `{edits: [{field, old, new}]}` → updated note; each edit appended to corrections.jsonl; note.json rewritten |
| `/api/sessions/{sid}/sign` | POST | `{lang: "en"|"hi"|"mr", doctor: {name, reg_no, clinic}}` → `{pdf_url}`; regenerates PDF signed (no DRAFT banner), state=signed |
| `/api/sessions/{sid}/pdf?lang=` | GET | → application/pdf (draft or signed, current note state) |
| `/api/sessions/{sid}/translate-advice` | POST | `{lang}` → `{advice}`; thin wrapper over /translate (kept for contract stability) |
| `/api/sessions/{sid}/translate` | POST | `{lang}` → `{note_values: {field_path: translated}, transcript: [per-turn translated]}`; batched Qwen (temp 0), cached in outputs/<sid>/translations_<lang>.json. NEVER translates medications[].*, vitals, or numbers (dosage safety). Translations are display-only — note.json stays source-language; edits always apply to originals |
| `/` | GET | index.html |

Browser recording: WebAudio → PCM → WAV blob client-side (NO MediaRecorder
webm — avoids server-side codec deps), POSTed to `/api/sessions`. A visible
"use audio file instead" input accepts wav/mp3 (the DONE demo uses
`data/sample_00.mp3`).

## Field paths (shared by PATCH edits, corrections.jsonl, provenance)
Dotted-with-index notation, exactly these forms:
`chief_complaint`, `history`, `examination`, `advice`, `follow_up`,
`symptoms[2].name`, `vitals[0].value`, `diagnosis[1].term`,
`medications[0].drug|dose|frequency|timing|duration`,
`investigations[3]`, `diagnostic_results[0]`.

## corrections.jsonl (append-only, one JSON object per line)
```json
{"ts": "2026-07-10T14:03:22+05:30", "field": "medications[0].dose", "old": null, "new": "650 mg", "lang": "en"}
```

## Flag translation (owner: backend/provenance.py — advisor-flagged trap)
`note.low_confidence_fields` are NAME-keyed strings ("medications.Dolo
650.dose_unknown", "chief_complaint"). The UI and PATCH work with INDEX-keyed
paths. `provenance.py` exports `flags_by_path(note) -> dict[path, reason]`
resolving names to indices (drug name → its row; bare field names map to the
whole-field path; unresolvable flags map to a top-level `_general` list, still
shown in the review footer). GET note returns this dict; the SPA never parses
raw flag strings.

## Ownership of shared modules
`web/translations.py` is owned by agent A (backend). Key set is fixed here:
`app_title, capture_hint, record, stop, processing, stage_transcribed,
stage_speakers, stage_drafting, review_title, patient, chief_complaint,
history, vitals, symptoms, examination, diagnosis, medications, drug, dose,
frequency, timing, duration, investigations, diagnostic_results, advice,
follow_up, verify, sign, signed, language, view_transcript, empty, doctor_name,
reg_no, clinic_address, draft_banner, rx`. Agents B and C consume it read-only
(B via /api/translations, C via import).

## Provenance (computed server-side at GET note; heuristic, honest)
For each populated leaf field: best transcript turn by token-overlap of the
field value vs turn text (fold Devanagari/Latin like eval/drug_bench._fold; ≥1
shared token ≥3 chars required). No match → provenance entry absent → UI shows
no provenance line (never fabricate). Response snippet = full turn text.

## Language
- UI labels: static table in `web/translations.py`, keys shared with frontend
  via `/api/translations` GET (dump of the dict) — zero latency after load.
- Free-text advice: translated on demand via local Qwen (route above), cached
  into note.json as `advice_translations: {hi: ..., mr: ...}`.
- Drug names ALWAYS Latin — structured fields are never machine-translated.
- PDF regenerates with the chosen language's labels; Devanagari text requires
  a bundled Unicode TTF (NotoSansDevanagari) registered in reportlab — vendored
  under `web/fonts/`, no runtime download.

## PDF (extends src/l5_render.py — one renderer, no fork)
`render(note, out_path=None, *, lang="en", signed=False, doctor=None)`
- signed=False: current DRAFT banner behavior.
- signed=True: banner replaced by signature block; regulatory placeholder block
  (doctor name, reg no., clinic address — from the sign request) in the header.
- Grayscale-safe: information never carried by color alone (⚑ glyph + text
  survive B&W); A5-printable margins.
- Labels via the same translations table (import from web.translations).

## DONE checklist (verified by coordinator, end-to-end on a real clip)
1. Upload data/sample_00.mp3 → process → review renders real note.
2. Edit one field → corrections.jsonl gains a structured diff line.
3. Transcript drawer opens; clicking ≥2 fields scrolls+highlights source lines.
4. Sign → PDF renders in en, hi, mr (labels translated, drugs Latin).

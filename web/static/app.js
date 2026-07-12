"use strict";

/* ---------- Silence auto-stop tuning (fix 3) ----------
 * RMS threshold calibrated ~0.01 full-scale for typical clinic mic gain;
 * SILENCE_STOP_SECONDS of continuous silence auto-stops the recording exactly
 * as if the user tapped stop; SILENCE_WARN_AT_SECONDS is when the quiet
 * countdown line appears (any speech above threshold cancels it). */
const SILENCE_RMS_THRESHOLD = 0.01;
const SILENCE_STOP_SECONDS = 28;
const SILENCE_WARN_AT_SECONDS = 20;

/* =====================================================================
 * MOCK LAYER — active only when opened directly as a file (no backend).
 * Keeps the whole two-screen flow demoable standalone. Everything below
 * this block until "END MOCK LAYER" is test fixture, not production code.
 * ===================================================================== */

window.MOCK = location.protocol === "file:";

const MOCK_TRANSLATIONS = {
  en: {
    app_title: "CliniScribe", capture_hint: "tap to record",
    record: "record", stop: "stop", processing: "processing",
    stage_transcribed: "transcribed", stage_speakers: "speakers separated",
    stage_drafting: "drafting note", review_title: "review",
    patient: "patient", chief_complaint: "chief complaint", history: "history",
    vitals: "vitals", symptoms: "symptoms", examination: "examination",
    diagnosis: "diagnosis", medications: "medications", drug: "drug",
    dose: "dose", frequency: "frequency", timing: "timing",
    duration: "duration", investigations: "investigations",
    diagnostic_results: "diagnostic results", advice: "advice",
    follow_up: "follow-up", verify: "verify", sign: "sign",
    signed: "signed", language: "language", view_transcript: "view transcript",
    empty: "—", doctor_name: "doctor name", reg_no: "reg. no.",
    clinic_address: "clinic address", draft_banner: "draft", rx: "rx",
  },
  hi: {
    app_title: "क्लिनिस्क्राइब", capture_hint: "रिकॉर्ड करने के लिए टैप करें",
    record: "रिकॉर्ड", stop: "रोकें", processing: "प्रसंस्करण",
    stage_transcribed: "ट्रांसक्राइब हुआ", stage_speakers: "वक्ता अलग हुए",
    stage_drafting: "नोट तैयार", review_title: "समीक्षा",
    patient: "रोगी", chief_complaint: "मुख्य शिकायत", history: "इतिहास",
    vitals: "महत्वपूर्ण संकेत", symptoms: "लक्षण", examination: "जांच",
    diagnosis: "निदान", medications: "दवाओं", drug: "दवा",
    dose: "मात्रा", frequency: "आवृत्ति", timing: "समय",
    duration: "अवधि", investigations: "जांचें",
    diagnostic_results: "जांच परिणाम", advice: "सलाह",
    follow_up: "फॉलो-अप", verify: "पुष्टि करें", sign: "हस्ताक्षर",
    signed: "हस्ताक्षरित", language: "भाषा", view_transcript: "ट्रांसक्रिप्ट देखें",
    empty: "—", doctor_name: "डॉक्टर का नाम", reg_no: "रजि. संख्या",
    clinic_address: "क्लिनिक पता", draft_banner: "ड्राफ्ट", rx: "प्रिस्क्रिप्शन",
  },
  mr: {
    app_title: "क्लिनिस्क्राइब", capture_hint: "रेकॉर्डिंगसाठी टॅप करा",
    record: "रेकॉर्ड", stop: "थांबा", processing: "प्रक्रिया",
    stage_transcribed: "लिप्यंतरित झाले", stage_speakers: "वक्ते वेगळे झाले",
    stage_drafting: "टीप तयार होत आहे", review_title: "पुनरावलोकन",
    patient: "रुग्ण", chief_complaint: "मुख्य तक्रार", history: "इतिहास",
    vitals: "महत्त्वाची लक्षणे", symptoms: "लक्षणे", examination: "तपासणी",
    diagnosis: "निदान", medications: "औषधे", drug: "औषध",
    dose: "मात्रा", frequency: "वारंवारता", timing: "वेळ",
    duration: "कालावधी", investigations: "चाचण्या",
    diagnostic_results: "चाचणी निकाल", advice: "सल्ला",
    follow_up: "पुनरभेट", verify: "पडताळा", sign: "सही",
    signed: "स्वाक्षरीत", language: "भाषा", view_transcript: "ट्रांसक्रिप्ट पहा",
    empty: "—", doctor_name: "डॉक्टरचे नाव", reg_no: "नोंद. क्र.",
    clinic_address: "दवाखान्याचा पत्ता", draft_banner: "ड्राफ्ट", rx: "प्रिस्क्रिप्शन",
  },
};

// Frontend-only microcopy NOT in the shared translations.py key set (English
// only — save/pdf-link controls, not clinical-note content).
const UI_STRINGS = {
  use_audio_file: "use audio file instead",
  save: "save",
  draft_pdf: "draft pdf and print",
  download_pdf: "download pdf",
  original: "original",
  editing_original: "editing original",
  live_preview_label: "LIVE PREVIEW — final transcript follows",
  // Background verification (fast + background check architecture).
  verify_running: "background check running…",
  verify_clear: "background check: all clear",
  verify_fields_prefix: "background check: ",
  verify_fields_suffix: " fields to review",
  verify_partial_suffix: " · partial check",
  verify_after_sign_prefix: "check completed after signing: ",
  verify_after_sign_suffix: " differences logged",
  use_checked_version: "use checked version",
  checked_transcript: "checked transcript",
  // /translate in-flight feedback (cold cache / loaded host can take 1-2 min).
  translating: "translating…",
  translation_unavailable: "translation unavailable — showing original",
};
function u(key) { return UI_STRINGS[key]; }

const MOCK_STATUS_SEQUENCE = [
  { state: "processing", stage: "l1_preprocess", stages_done: [], error: null },
  { state: "processing", stage: "l2_diarize", stages_done: ["l1_preprocess"], error: null },
  { state: "processing", stage: "l3_asr", stages_done: ["l1_preprocess", "l2_diarize"], error: null },
  { state: "processing", stage: "l3_5_normalize", stages_done: ["l1_preprocess", "l2_diarize", "l3_asr"], error: null },
  { state: "processing", stage: "l4_extract", stages_done: ["l1_preprocess", "l2_diarize", "l3_asr", "l3_5_normalize"], error: null },
  { state: "review", stage: null, stages_done: ["l1_preprocess", "l2_diarize", "l3_asr", "l3_5_normalize", "l4_extract"], error: null },
];

const MOCK_NOTE = {
  chief_complaint: "Fever and cough for 3 days",
  history: "No known drug allergies. Similar episode last winter.",
  symptoms: [
    { name: "Fever", finding_status: "Present", severity: "Moderate", since: "3 days" },
    { name: "Cough", finding_status: "Present", severity: "Mild", since: "3 days" },
    { name: "Headache", finding_status: "Absent", severity: null, since: null },
  ],
  vitals: [
    { name: "BP", value: "120/80 mmHg" },
    { name: "Temperature", value: "100.4 F" },
    { name: "SpO2", value: "98 %" },
  ],
  examination: "Throat mildly congested, chest clear on auscultation.",
  diagnosis: [
    { term: "Viral upper respiratory tract infection", snomed_id: "54150009", status: "Suspected" },
  ],
  medications: [
    { drug: "Paracetamol", dose: "650 mg", frequency: "twice daily", timing: "after food", duration: "3 days", validated: true },
    { drug: "Azithral", dose: null, frequency: "once daily", timing: "after food", duration: "5 days", validated: false },
  ],
  investigations: ["CBC", "Chest X-ray"],
  diagnostic_results: [],
  advice: "Drink plenty of fluids and take rest.",
  follow_up: "Return after 5 days if fever persists",
  low_confidence_fields: ["medications.Azithral.dose_unknown", "medications.Azithral.unvalidated", "vitals"],
};

const MOCK_TRANSCRIPT = [
  { speaker_role: "DOCTOR", text: "Namaste, what brings you in today?", start: 0.0, end: 3.2 },
  { speaker_role: "PATIENT", text: "Doctor, I have fever and cough since 3 days.", start: 3.4, end: 7.1 },
  { speaker_role: "DOCTOR", text: "Okay, blood pressure is 120 over 80, temperature 100.4 fahrenheit, SpO2 98 percent.", start: 7.5, end: 14.0 },
  { speaker_role: "DOCTOR", text: "Throat looks mildly congested, chest is clear.", start: 14.2, end: 18.0 },
  { speaker_role: "DOCTOR", text: "This looks like a viral upper respiratory infection, suspected.", start: 18.3, end: 23.0 },
  { speaker_role: "DOCTOR", text: "I'll prescribe paracetamol 650 mg twice daily after food for 3 days.", start: 23.4, end: 29.0 },
  { speaker_role: "DOCTOR", text: "Also azithral once daily after food for 5 days.", start: 29.2, end: 34.0 },
  { speaker_role: "DOCTOR", text: "Get a CBC and chest x-ray done. Drink plenty of fluids and take rest.", start: 34.3, end: 40.0 },
  { speaker_role: "DOCTOR", text: "Come back after 5 days if fever persists.", start: 40.2, end: 44.0 },
];

const MOCK_PROVENANCE = {
  "chief_complaint": { turn_index: 1, start: 3.4, end: 7.1, snippet: MOCK_TRANSCRIPT[1].text },
  "symptoms[0].name": { turn_index: 1, start: 3.4, end: 7.1, snippet: MOCK_TRANSCRIPT[1].text },
  "symptoms[1].name": { turn_index: 1, start: 3.4, end: 7.1, snippet: MOCK_TRANSCRIPT[1].text },
  "vitals[0].value": { turn_index: 2, start: 7.5, end: 14.0, snippet: MOCK_TRANSCRIPT[2].text },
  "vitals[1].value": { turn_index: 2, start: 7.5, end: 14.0, snippet: MOCK_TRANSCRIPT[2].text },
  "vitals[2].value": { turn_index: 2, start: 7.5, end: 14.0, snippet: MOCK_TRANSCRIPT[2].text },
  "examination": { turn_index: 3, start: 14.2, end: 18.0, snippet: MOCK_TRANSCRIPT[3].text },
  "diagnosis[0].term": { turn_index: 4, start: 18.3, end: 23.0, snippet: MOCK_TRANSCRIPT[4].text },
  "medications[0].drug": { turn_index: 5, start: 23.4, end: 29.0, snippet: MOCK_TRANSCRIPT[5].text },
  "medications[0].dose": { turn_index: 5, start: 23.4, end: 29.0, snippet: MOCK_TRANSCRIPT[5].text },
  "medications[0].frequency": { turn_index: 5, start: 23.4, end: 29.0, snippet: MOCK_TRANSCRIPT[5].text },
  "medications[0].timing": { turn_index: 5, start: 23.4, end: 29.0, snippet: MOCK_TRANSCRIPT[5].text },
  "medications[0].duration": { turn_index: 5, start: 23.4, end: 29.0, snippet: MOCK_TRANSCRIPT[5].text },
  "medications[1].drug": { turn_index: 6, start: 29.2, end: 34.0, snippet: MOCK_TRANSCRIPT[6].text },
  "investigations[0]": { turn_index: 7, start: 34.3, end: 40.0, snippet: MOCK_TRANSCRIPT[7].text },
  "investigations[1]": { turn_index: 7, start: 34.3, end: 40.0, snippet: MOCK_TRANSCRIPT[7].text },
  "advice": { turn_index: 7, start: 34.3, end: 40.0, snippet: MOCK_TRANSCRIPT[7].text },
  "follow_up": { turn_index: 8, start: 40.2, end: 44.0, snippet: MOCK_TRANSCRIPT[8].text },
};

const MOCK_FLAGS = {
  "medications[1].dose": "Dose for Azithral could not be confirmed — verify with the patient",
  "medications[1].drug": "'Azithral' is not in the CDSCO drug list — verify the drug name",
  "_general": ["Vital sign could not be confirmed — re-measure if needed"],
};

let mockStatusTick = 0;
let mockNoteState = JSON.parse(JSON.stringify(MOCK_NOTE));
let mockTranslationsCache = {}; // { lang: {note_values, transcript} }

// Mirrors web/app.py's _translatable_note_values: fields eligible for
// display-only translation (medications and vitals are never translated).
function mockTranslatableNoteValues(note) {
  const values = {};
  for (const key of ["chief_complaint", "history", "examination", "advice", "follow_up"]) {
    if (note[key]) values[key] = note[key];
  }
  (note.symptoms || []).forEach((s, i) => { if (s.name) values[`symptoms[${i}].name`] = s.name; });
  (note.diagnosis || []).forEach((d, i) => { if (d.term) values[`diagnosis[${i}].term`] = d.term; });
  (note.investigations || []).forEach((v, i) => { if (v) values[`investigations[${i}]`] = v; });
  (note.diagnostic_results || []).forEach((v, i) => { if (v) values[`diagnostic_results[${i}]`] = v; });
  return values;
}

function mockJsonResponse(body) {
  return Promise.resolve(new Response(JSON.stringify(body), {
    status: 200, headers: { "Content-Type": "application/json" },
  }));
}

function mockFetch(url, opts = {}) {
  const method = (opts.method || "GET").toUpperCase();

  if (url === "/api/translations") {
    return mockJsonResponse(MOCK_TRANSLATIONS);
  }
  if (url === "/api/sessions" && method === "POST") {
    return mockJsonResponse({ session_id: "mock-session" });
  }
  if (/\/api\/sessions\/[^/]+\/process$/.test(url) && method === "POST") {
    mockStatusTick = 0;
    return mockJsonResponse({ session_id: "mock-session" });
  }
  if (/\/api\/sessions\/[^/]+\/status$/.test(url)) {
    const idx = Math.min(mockStatusTick, MOCK_STATUS_SEQUENCE.length - 1);
    mockStatusTick += 1;
    return mockJsonResponse(MOCK_STATUS_SEQUENCE[idx]);
  }
  if (/\/api\/sessions\/[^/]+\/note$/.test(url) && method === "GET") {
    return mockJsonResponse({
      note: mockNoteState, transcript: MOCK_TRANSCRIPT,
      provenance: MOCK_PROVENANCE, flags: MOCK_FLAGS,
    });
  }
  if (/\/api\/sessions\/[^/]+\/note$/.test(url) && method === "PATCH") {
    const body = JSON.parse(opts.body || "{}");
    for (const edit of body.edits || []) {
      setValueAtPath(mockNoteState, edit.field, edit.new);
    }
    return mockJsonResponse({ note: mockNoteState });
  }
  if (/\/api\/sessions\/[^/]+\/translate$/.test(url) && method === "POST") {
    const body = JSON.parse(opts.body || "{}");
    const lang = body.lang;
    if (lang === "en") return mockJsonResponse({ note_values: {}, transcript: [] });
    if (!mockTranslationsCache[lang]) {
      const note_values = {};
      for (const [path, text] of Object.entries(mockTranslatableNoteValues(mockNoteState))) {
        note_values[path] = `[${lang}] ${text}`;
      }
      const transcript = MOCK_TRANSCRIPT.map((turn) => `[${lang}] ${turn.text}`);
      mockTranslationsCache[lang] = { note_values, transcript };
    }
    return mockJsonResponse(mockTranslationsCache[lang]);
  }
  if (/\/api\/sessions\/[^/]+\/sign$/.test(url) && method === "POST") {
    return mockJsonResponse({ pdf_url: "about:blank" });
  }
  if (/\/api\/sessions\/[^/]+\/pdf/.test(url)) {
    return mockJsonResponse({ note: "mock: no real PDF in offline demo" });
  }
  return Promise.reject(new Error(`mock: unhandled route ${method} ${url}`));
}

/* ===================== END MOCK LAYER ===================== */

function api(url, opts) {
  return (window.MOCK ? mockFetch : fetch)(url, opts);
}

/* =====================================================================
 * State
 * ===================================================================== */

let sessionId = null;
let currentLang = "en";
let translations = {}; // { en: {...}, hi: {...}, mr: {...} }
let statusPollHandle = null;
let recordStartTime = null;
let timerHandle = null;

let noteData = null;       // ClinicalNote-shaped object
let provenanceData = {};   // { path: {turn_index, start, end, snippet} }
let flagsData = {};        // { path: reason } + optional _general: [reasons]
let transcriptData = [];   // [{speaker_role, text, start, end}]
let baselineValues = {};   // { path: string } — value at last load/save, for diffing edits
let pendingEdits = {};     // { path: {field, old, new} }

// Display-only machine translation (fix 4). Values on disk (noteData,
// transcriptData) always stay source-language; these hold the hi/mr overlay.
// NEVER populated for medications[].*/vitals[].value — dosage/numeric safety.
let translatedValues = {};      // { path: translated text }
let translatedTranscript = [];  // [translated text] aligned to transcriptData
let translationsByLang = {};    // client-side cache: { lang: {note_values, transcript} }

// Latency fix: the initial auto-translate (loadNoteAndShowReview) now runs
// in the BACKGROUND after first paint instead of blocking it. This flags a
// re-render that arrived while the reviewer was mid-edit in an inline
// contenteditable field, so it can be deferred (applied on the field's next
// blur) instead of clobbering in-progress, unsaved keystrokes.
let deferredReviewRerenderPending = false;

// Background verification (fast + background check architecture): the fast
// engine's note is reviewable immediately; a daemon-thread "second listen"
// (web/verification.py) re-checks safety-critical fields and reports here.
let verificationData = null;       // full GET .../verification body, once fetched
let lastVerificationState = null;  // dedupes re-fetching on every 1s status poll
let lastVerificationStatusObj = null;  // last {state, n_differing, coverage} from status.json
let showCheckedTranscript = false; // drawer toggle: original transcript vs. checked spans

function t(key) {
  const table = translations[currentLang] || {};
  return table[key] ?? (translations.en || {})[key] ?? key;
}

/* =====================================================================
 * Field-path helpers (dotted-with-index notation, spec Field paths section)
 * ===================================================================== */

const PATH_RE = /^([a-zA-Z_]+)(?:\[(\d+)\])?(?:\.([a-zA-Z_]+))?$/;

function getValueAtPath(note, path) {
  const m = path.match(PATH_RE);
  if (!m) return undefined;
  const [, key, idx, sub] = m;
  let v = note[key];
  if (idx !== undefined) v = v?.[Number(idx)];
  if (sub !== undefined) v = v?.[sub];
  return v;
}

function setValueAtPath(note, path, newVal) {
  const m = path.match(PATH_RE);
  if (!m) return;
  const [, key, idx, sub] = m;
  if (idx === undefined) {
    note[key] = newVal;
    return;
  }
  const item = note[key][Number(idx)];
  if (sub === undefined) {
    note[key][Number(idx)] = newVal;
  } else {
    item[sub] = newVal;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function formatTs(start, end) {
  return `[${start.toFixed(1)}–${end.toFixed(1)}]s`;
}

/* =====================================================================
 * Screen 1 — Capture
 * ===================================================================== */

const micRing = document.getElementById("mic-ring");
const captureCaption = document.getElementById("capture-caption");
const timerEl = document.getElementById("timer");
const silenceNoticeEl = document.getElementById("silence-notice");
const stageLinesEl = document.getElementById("stage-lines");
const fileInput = document.getElementById("file-input");
const fileHintText = document.getElementById("file-hint-text");
const livePreviewEl = document.getElementById("live-preview");
const livePreviewLabelEl = document.getElementById("live-preview-label");
const livePreviewTextEl = document.getElementById("live-preview-text");

let captureState = "idle"; // idle | recording | uploading | processing

let mediaStream = null;
let audioContext = null;
let processorNode = null;
let sourceNode = null;
let recordedChunks = []; // Float32Array chunks at 16kHz mono

let silenceStartedAt = null; // ms timestamp when continuous silence began, or null
let autoStopping = false;    // guards against re-entrant auto-stop while stopping

/* ---------- Live in-recording transcript preview (UI-display-only) ----------
 * Every LIVE_PREVIEW_INTERVAL_MS, POST the full recorded-so-far WAV to
 * /api/live/preview and render the returned running text. Gate-exempt: this
 * text NEVER feeds the note or the review screen (web/live_asr.py contract).
 * Graceful degradation: any failure (mlx_whisper missing, 429 debounce, a
 * transient network error) is swallowed silently — capture must work with
 * or without the preview. */
const LIVE_PREVIEW_INTERVAL_MS = 10000;
let livePreviewTimerHandle = null;
let livePreviewInFlight = false; // client-side debounce, mirrors the server's

function rms(buffer) {
  let sumSquares = 0;
  for (let i = 0; i < buffer.length; i++) sumSquares += buffer[i] * buffer[i];
  return Math.sqrt(sumSquares / buffer.length);
}

function showSilenceCountdown(secondsRemaining) {
  silenceNoticeEl.hidden = false;
  silenceNoticeEl.textContent = `silence — auto-stop in ${secondsRemaining}s`;
}

function hideSilenceCountdown() {
  silenceNoticeEl.hidden = true;
}

function checkSilenceAndMaybeAutoStop(inputBuffer) {
  if (autoStopping) return;
  if (rms(inputBuffer) < SILENCE_RMS_THRESHOLD) {
    if (silenceStartedAt === null) silenceStartedAt = Date.now();
    const elapsed = (Date.now() - silenceStartedAt) / 1000;
    if (elapsed >= SILENCE_STOP_SECONDS) {
      autoStopping = true;
      hideSilenceCountdown();
      stopRecordingAndUpload().catch((err) => console.error("upload failed:", err));
    } else if (elapsed >= SILENCE_WARN_AT_SECONDS) {
      showSilenceCountdown(Math.ceil(SILENCE_STOP_SECONDS - elapsed));
    }
  } else {
    // Any speech above threshold cancels the pending auto-stop.
    silenceStartedAt = null;
    hideSilenceCountdown();
  }
}

function downsampleBuffer(buffer, inputRate, outputRate) {
  if (outputRate === inputRate) return buffer;
  const ratio = inputRate / outputRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < newLength) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([view], { type: "audio/wav" });
}

const TARGET_SAMPLE_RATE = 16000;

function buildCurrentWavBlob() {
  const totalLength = recordedChunks.reduce((n, c) => n + c.length, 0);
  const samples = new Float32Array(totalLength);
  let off = 0;
  for (const chunk of recordedChunks) { samples.set(chunk, off); off += chunk.length; }
  return encodeWav(samples, TARGET_SAMPLE_RATE);
}

function renderLivePreview(text) {
  if (!text) return;
  livePreviewTextEl.textContent = text;
  livePreviewEl.hidden = false;
  livePreviewTextEl.scrollTop = livePreviewTextEl.scrollHeight; // newest stays visible
}

async function pollLivePreview() {
  if (livePreviewInFlight || !recordedChunks.length) return;
  livePreviewInFlight = true;
  try {
    const form = new FormData();
    form.append("audio", buildCurrentWavBlob(), "live.wav");
    const res = await api("/api/live/preview", { method: "POST", body: form });
    if (res.ok) {
      const data = await res.json();
      renderLivePreview(data.text);
    }
  } catch (err) {
    // Graceful degradation (spec): live preview never surfaces a failure —
    // capture must keep working with or without mlx-whisper installed.
  } finally {
    livePreviewInFlight = false;
  }
}

async function startRecording() {
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  processorNode = audioContext.createScriptProcessor(4096, 1, 1);
  recordedChunks = [];

  processorNode.onaudioprocess = (e) => {
    const input = e.inputBuffer.getChannelData(0);
    const down = downsampleBuffer(input, audioContext.sampleRate, TARGET_SAMPLE_RATE);
    recordedChunks.push(new Float32Array(down));
    checkSilenceAndMaybeAutoStop(input);
  };

  sourceNode.connect(processorNode);
  processorNode.connect(audioContext.destination);

  captureState = "recording";
  micRing.classList.add("recording");
  micRing.setAttribute("aria-label", t("stop"));
  captureCaption.hidden = true;
  timerEl.hidden = false;
  recordStartTime = Date.now();
  timerHandle = setInterval(updateTimer, 200);
  silenceStartedAt = null;
  autoStopping = false;
  hideSilenceCountdown();

  livePreviewTextEl.textContent = "";
  livePreviewEl.hidden = true;
  livePreviewTimerHandle = setInterval(pollLivePreview, LIVE_PREVIEW_INTERVAL_MS);
}

function updateTimer() {
  const elapsed = Math.floor((Date.now() - recordStartTime) / 1000);
  const mm = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const ss = String(elapsed % 60).padStart(2, "0");
  timerEl.textContent = `${mm}:${ss}`;
}

function stopRecordingTracks() {
  if (processorNode) { processorNode.disconnect(); processorNode = null; }
  if (sourceNode) { sourceNode.disconnect(); sourceNode = null; }
  if (mediaStream) { mediaStream.getTracks().forEach((tr) => tr.stop()); mediaStream = null; }
  if (audioContext) { audioContext.close(); audioContext = null; }
  clearInterval(timerHandle);
  clearInterval(livePreviewTimerHandle);
  livePreviewTimerHandle = null;
  livePreviewEl.hidden = true;
}

async function stopRecordingAndUpload() {
  stopRecordingTracks();
  hideSilenceCountdown();
  silenceStartedAt = null;
  micRing.classList.remove("recording");
  micRing.setAttribute("aria-label", t("record"));

  const totalLength = recordedChunks.reduce((n, c) => n + c.length, 0);
  const samples = new Float32Array(totalLength);
  let off = 0;
  for (const chunk of recordedChunks) { samples.set(chunk, off); off += chunk.length; }
  const wavBlob = encodeWav(samples, TARGET_SAMPLE_RATE);

  await uploadAndProcess(wavBlob, "recording.wav");
}

async function uploadAndProcess(blob, filename) {
  captureState = "uploading";
  micRing.classList.add("busy");

  const form = new FormData();
  form.append("audio", blob, filename);
  const res = await api("/api/sessions", { method: "POST", body: form });
  const data = await res.json();
  sessionId = data.session_id;

  await api(`/api/sessions/${sessionId}/process`, { method: "POST" });

  captureState = "processing";
  stageLinesEl.hidden = false;
  startStatusPolling();
}

function setStageLine(stageKey, state) {
  const el = stageLinesEl.querySelector(`[data-stage="${stageKey}"]`);
  el.classList.remove("done", "active");
  if (state) el.classList.add(state);
}

// %/ETA suffix, on whichever stage-line is active (spec: appears the instant
// a calibrated stage starts, disappears the moment it ends).
function formatEtaSuffix(etaSeconds) {
  if (typeof etaSeconds !== "number") return "";
  const rounded = etaSeconds < 60
    ? `~${Math.round(etaSeconds)}s left`
    : `~${Math.round(etaSeconds / 60)} min left`;
  return ` · ${rounded}`;
}

/* ---------- Calibrated + real-time stage progress (issue 1) ----------
 * Three DOM stage-lines (index.html: data-stage="transcribed"/"speakers"/
 * "drafting") map to four BACKEND stage names that can carry a %/ETA suffix.
 * l1_preprocess is intentionally excluded: it is fast and low-variance
 * (never had a %/ETA display even before this fix — see web/app.py's
 * calibration comment) and is not one of the four stages the server
 * calibrates.
 *
 * l3_asr is the ONLY stage with a real per-window progress callback
 * (src/fast_asr.py's on_progress, mirrored into status.json's
 * "progress"/"eta_seconds"). l2_diarize, l3_5_normalize, and l4_extract have
 * no internal callback at all — calibrated interpolation against the
 * server's expected_stage_seconds is the whole story for them, which is
 * honest because they are short, low-variance stages where a calibrated
 * median is a reasonable stand-in for "how far along am I".
 *
 * Instantness is bounded by the status-poll cadence (500ms-1s): a stage's
 * suffix can only appear once the SPA has polled and observed that stage is
 * active. There is no push channel here (out of scope) to do better than
 * that. Once observed, though, the suffix itself is not poll-cadence-limited
 * — a separate, faster interval (renderSmoothProgress) advances the display
 * every SMOOTH_PROGRESS_INTERVAL_MS by wall-clock time alone, so it never
 * visibly freezes between polls. */
const STAGE_TO_LINE = {
  l2_diarize: "transcribed",
  l3_asr: "transcribed",
  l3_5_normalize: "speakers",
  l4_extract: "drafting",
};

// Client-side fallback, used only if the server never wrote
// status.expected_stage_seconds (e.g. it could not estimate the upload's
// audio duration) — rough seconds, NOT audio-scaled, just enough to keep the
// display honest-ish rather than blank.
const FALLBACK_STAGE_SECONDS = { l2_diarize: 10, l3_asr: 15, l3_5_normalize: 18, l4_extract: 20 };

const SMOOTH_PROGRESS_INTERVAL_MS = 200;

let expectedStageSeconds = {};  // status.expected_stage_seconds, cached once per session
let stageProgressModel = null;  // { stage, clientStart, expectedSeconds, real: {...}|null }
let smoothProgressHandle = null;

function updateStageProgressFromStatus(status) {
  if (status.expected_stage_seconds) expectedStageSeconds = status.expected_stage_seconds;

  const stage = STAGE_TO_LINE[status.stage] ? status.stage : null;
  if (!stage) {
    stageProgressModel = null;
    return;
  }
  if (!stageProgressModel || stageProgressModel.stage !== stage) {
    stageProgressModel = {
      stage,
      clientStart: Date.now(),
      expectedSeconds: expectedStageSeconds[stage] || FALLBACK_STAGE_SECONDS[stage] || 15,
      real: null,
    };
  }
  if (stage === "l3_asr" && typeof status.progress === "number") {
    stageProgressModel.real = {
      progress: status.progress,
      etaSeconds: typeof status.eta_seconds === "number" ? status.eta_seconds : 0,
      capturedAtMs: Date.now(),
    };
  }
}

// Blend rule: a real progress/eta datum (l3_asr only) always wins the moment
// it arrives; between polls, both the real-anchored model and the
// calibration-only model advance by wall-clock time alone. Both branches
// compute the SAME two numbers (pct, etaSeconds) from one implied total
// duration, so they can never visibly disagree with each other.
function currentStageDisplay(nowMs) {
  const model = stageProgressModel;
  if (!model) return null;

  let totalExpected, elapsedTotal;
  if (model.real) {
    const { progress, etaSeconds, capturedAtMs } = model.real;
    const dt = (nowMs - capturedAtMs) / 1000;
    totalExpected = progress < 1 ? etaSeconds / (1 - progress) : (nowMs - model.clientStart) / 1000;
    elapsedTotal = totalExpected - etaSeconds + dt;
  } else {
    totalExpected = model.expectedSeconds;
    elapsedTotal = (nowMs - model.clientStart) / 1000;
  }
  if (!(totalExpected > 0)) return null;

  const pct = Math.min(0.95, Math.max(0, elapsedTotal / totalExpected));
  return { line: STAGE_TO_LINE[model.stage], pct, etaSeconds: totalExpected * (1 - pct) };
}

function renderSmoothProgress() {
  const display = currentStageDisplay(Date.now());
  for (const line of ["transcribed", "speakers", "drafting"]) {
    const el = stageLinesEl.querySelector(`[data-stage="${line}"] .progress`);
    if (display && display.line === line) {
      el.textContent = ` · ${Math.round(display.pct * 100)}%${formatEtaSuffix(display.etaSeconds)}`;
      el.hidden = false;
    } else {
      el.hidden = true;
      el.textContent = "";
    }
  }
}

function applyStatus(status) {
  const done = new Set(status.stages_done || []);
  const stage = status.stage;

  const transcribedDone = done.has("l1_preprocess") && done.has("l2_diarize") && done.has("l3_asr");
  setStageLine("transcribed", transcribedDone ? "done" : (stage && ["l1_preprocess", "l2_diarize", "l3_asr"].includes(stage) ? "active" : ""));

  const speakersDone = done.has("l3_5_normalize");
  setStageLine("speakers", speakersDone ? "done" : (stage === "l3_5_normalize" ? "active" : ""));

  const draftingDone = done.has("l4_extract");
  setStageLine("drafting", draftingDone ? "done" : (stage === "l4_extract" ? "active" : ""));

  updateStageProgressFromStatus(status);
}

// (d) Background verification is polled on this SAME status timer — no new
// timer. Once "review" is reached the SPA keeps polling (instead of
// stopping, as it did before verification existed) so it can learn when a
// fast-engine session's background check finishes; it stops once
// verification reaches a terminal state, or after a short grace window if
// this session never gets a "verification" key at all (the accurate engine
// never starts one — see web/app.py's _run_pipeline_thread).
const VERIFICATION_POLL_GRACE_TICKS = 5;
let reviewLoaded = false;
let verificationPollGraceTicks = 0;

// Issue 2: poll faster once L4/L5 are in play, so the SPA notices "review"
// (and the note itself, via maybePrefetchNote below) sooner than the
// standard 1s cadence — cheap, since these are the LAST couple of stages.
const STATUS_POLL_NORMAL_MS = 1000;
const STATUS_POLL_FAST_MS = 500;
let pollingActive = false;

function isFastPollPhase(status) {
  const done = status.stages_done || [];
  return status.stage === "l4_extract" || status.stage === "l5_render" || done.includes("l4_extract");
}

// Issue 2: kick off GET .../note as soon as L4 has finished — well before
// status flips to "review" (L5 render + note_meta precompute + a poll tick
// still stand between the two) — so the note is often already in hand by
// the time the SPA notices "review".
let notePrefetchPromise = null;
let notePrefetchStartedForSid = null;

async function fetchNoteJson() {
  const res = await api(`/api/sessions/${sessionId}/note`);
  if (!res.ok) throw new Error(`note fetch failed: ${res.status}`);
  return res.json();
}

function maybePrefetchNote(status) {
  if (reviewLoaded) return;
  if (!(status.stages_done || []).includes("l4_extract")) return;
  if (notePrefetchStartedForSid === sessionId) return;  // already in flight or resolved
  notePrefetchStartedForSid = sessionId;
  notePrefetchPromise = fetchNoteJson().catch((err) => {
    // Likely a tight race with L5/note_meta not written yet -- clear the
    // guard so the next poll tick (still the 500ms "fast" phase) retries.
    notePrefetchStartedForSid = null;
    throw err;
  });
}

function stopStatusPolling() {
  pollingActive = false;
  if (smoothProgressHandle) { clearInterval(smoothProgressHandle); smoothProgressHandle = null; }
  stageProgressModel = null;
  renderSmoothProgress();
}

async function runStatusPoll() {
  if (!pollingActive) return;
  let nextDelayMs = STATUS_POLL_NORMAL_MS;
  // A transient 500 (e.g. a status.json write in progress server-side) must
  // never break polling — swallow and silently retry next tick.
  try {
    const res = await api(`/api/sessions/${sessionId}/status`);
    if (res.ok) {
      const status = await res.json();
      applyStatus(status);
      if (isFastPollPhase(status)) nextDelayMs = STATUS_POLL_FAST_MS;
      maybePrefetchNote(status);
      if (status.state === "review" && !reviewLoaded) {
        reviewLoaded = true;
        await loadNoteAndShowReview();
      }
      if (reviewLoaded || status.state === "signed") {
        if (status.verification) {
          handleVerificationStatus(status.verification);
          if (status.verification.state === "done" || status.verification.state === "failed") {
            stopStatusPolling();
            return;
          }
        } else {
          verificationPollGraceTicks += 1;
          if (verificationPollGraceTicks > VERIFICATION_POLL_GRACE_TICKS) {
            stopStatusPolling();
            return;
          }
        }
      }
      // status.error is surfaced only via console — no dashboard element per spec.
      if (status.error) console.error("pipeline error:", status.error);
    }
  } catch (err) {
    console.error("status poll failed, retrying next tick:", err);
  }
  if (pollingActive) statusPollHandle = setTimeout(runStatusPoll, nextDelayMs);
}

function startStatusPolling() {
  applyStatus({ state: "processing", stage: null, stages_done: [] });
  reviewLoaded = false;
  verificationPollGraceTicks = 0;
  expectedStageSeconds = {};
  stageProgressModel = null;
  notePrefetchPromise = null;
  notePrefetchStartedForSid = null;
  pollingActive = true;
  smoothProgressHandle = setInterval(renderSmoothProgress, SMOOTH_PROGRESS_INTERVAL_MS);
  statusPollHandle = setTimeout(runStatusPoll, STATUS_POLL_NORMAL_MS);
}

micRing.addEventListener("click", () => {
  if (captureState === "idle") {
    startRecording().catch((err) => console.error("mic access failed:", err));
  } else if (captureState === "recording") {
    stopRecordingAndUpload().catch((err) => console.error("upload failed:", err));
  }
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  stageLinesEl.hidden = false;
  micRing.classList.add("busy");
  uploadAndProcess(file, file.name).catch((err) => console.error("upload failed:", err));
});

/* =====================================================================
 * Screen switching
 * ===================================================================== */

const screenCapture = document.getElementById("screen-capture");
const screenReview = document.getElementById("screen-review");

function showScreen(name) {
  screenCapture.classList.toggle("active", name === "capture");
  screenReview.classList.toggle("active", name === "review");
}

/* =====================================================================
 * Screen 2 — Review
 * ===================================================================== */

const noteGrid = document.getElementById("note-grid");
const generalFlagsEl = document.getElementById("general-flags");
const drawerEl = document.getElementById("drawer");
const drawerTurnsEl = document.getElementById("drawer-turns");
const verificationBannerEl = document.getElementById("verification-banner");
const checkedTranscriptToggleEl = document.getElementById("checked-transcript-toggle");

// Bug 1 (target-language-absolute): cheap client-side check for whether the
// note/transcript carries any non-Latin (Devanagari or Arabic) script, so a
// note-load can immediately fire a translation to the (default "en")
// selected language when the two mismatch, without a wasted round trip when
// the note is already Latin-script. Same script ranges as web/app.py's
// _needs_translation_to_en / src/l5_render.py's _SCRIPT_RUN_RE.
const NON_LATIN_SCRIPT_RE = /[ऀ-ॿ؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]/;

function noteHasNonLatinScript(note, turns) {
  const texts = [];
  for (const key of ["chief_complaint", "history", "examination", "advice", "follow_up"]) {
    if (note[key]) texts.push(note[key]);
  }
  (note.symptoms || []).forEach((s) => { if (s.name) texts.push(s.name); });
  (note.diagnosis || []).forEach((d) => { if (d.term) texts.push(d.term); });
  (note.investigations || []).forEach((v) => { if (v) texts.push(v); });
  (note.diagnostic_results || []).forEach((v) => { if (v) texts.push(v); });
  (turns || []).forEach((turn) => { if (turn.text) texts.push(turn.text); });
  return texts.some((text) => NON_LATIN_SCRIPT_RE.test(text));
}

/* =====================================================================
 * Background verification (fast + background check architecture)
 * ===================================================================== */

function isSignedNow() {
  return !signedPanel.hidden;
}

// Three banner states (spec): running / all clear / N fields to review, plus
// a fourth, read-only wording once the doctor has already signed — the note
// is final at that point, so this is informational only, never actionable.
function verificationBannerText(verification, signed) {
  if (!verification) return null;
  const n = verification.n_differing || 0;
  const partialSuffix = verification.coverage === "partial" ? u("verify_partial_suffix") : "";
  if (signed) {
    if (verification.state !== "done") return null;
    return `${u("verify_after_sign_prefix")}${n}${u("verify_after_sign_suffix")}`;
  }
  if (verification.state === "running") return u("verify_running");
  if (verification.state === "done") {
    return n === 0
      ? `${u("verify_clear")} ✓${partialSuffix}`
      : `${u("verify_fields_prefix")}${n}${u("verify_fields_suffix")}${partialSuffix}`;
  }
  return null; // "failed": no banner line, per spec's three live states
}

function updateVerificationBanner(verification) {
  const text = verificationBannerText(verification, isSignedNow());
  if (!text) {
    verificationBannerEl.hidden = true;
    return;
  }
  verificationBannerEl.hidden = false;
  verificationBannerEl.textContent = text;
  verificationBannerEl.classList.toggle(
    "attention",
    !isSignedNow() && verification.state === "done" && (verification.n_differing || 0) > 0
  );
}

function updateCheckedTranscriptToggle() {
  const available = !!(verificationData && (verificationData.transcript_accurate || []).length);
  checkedTranscriptToggleEl.hidden = !available;
  checkedTranscriptToggleEl.textContent = u("checked_transcript");
  checkedTranscriptToggleEl.classList.toggle("active", showCheckedTranscript);
}

async function fetchVerificationDetails() {
  try {
    const res = await api(`/api/sessions/${sessionId}/verification`);
    if (!res.ok) return;
    verificationData = await res.json();
  } catch (err) {
    console.error("verification fetch failed:", err);
    return;
  }
  updateCheckedTranscriptToggle();
  if (screenReview.classList.contains("active")) {
    renderReview();
    renderDrawer();
  }
}

// Called on every status poll tick once the review screen has loaded (or the
// session is signed) — (d) "poll verification state on the same status poll,
// no new timer". Fetches the full diff/transcript detail only once, the
// first time verification reaches a terminal state.
function handleVerificationStatus(verification) {
  lastVerificationStatusObj = verification;
  updateVerificationBanner(verification);
  if (!verification || verification.state === lastVerificationState) return;
  lastVerificationState = verification.state;
  if (verification.state === "done" || verification.state === "failed") {
    fetchVerificationDetails();
  }
}

// A field is flagged if either the original L4 note flags it OR background
// verification found a difference — but never after signing (note.json is
// final; no more flags-editing, per the sign-interplay rule).
function verificationDiffFor(path) {
  if (!verificationData || isSignedNow()) return null;
  return (verificationData.fields_differing || []).find((d) => d.field === path) || null;
}

function isFlagged(path) {
  return !!flagsData[path] || !!verificationDiffFor(path);
}

async function applyCheckedVersion(path) {
  const diff = verificationDiffFor(path);
  if (!diff) return;
  const oldVal = baselineValues[path] === "" ? null : baselineValues[path];
  const newVal = diff.accurate_value;
  const res = await api(`/api/sessions/${sessionId}/note`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ edits: [{ field: path, old: oldVal, new: newVal }] }),
  });
  const data = await res.json();
  noteData = data.note;
  baselineValues[path] = newVal ?? "";
  delete pendingEdits[path];
  if (verificationData) {
    verificationData.fields_differing = (verificationData.fields_differing || []).filter(
      (d) => d.field !== path
    );
  }
  renderReview();
}

async function loadNoteAndShowReview() {
  // Issue 2: reuse the in-flight/already-resolved prefetch kicked off by
  // maybePrefetchNote as soon as l4_extract finished, instead of always
  // paying for a fresh GET .../note round trip here. A prefetch failure
  // (e.g. the tight race before L5/note_meta finished writing) falls back to
  // a normal fetch, exactly as before this change.
  let data = null;
  if (notePrefetchPromise) {
    try {
      data = await notePrefetchPromise;
    } catch (err) {
      data = null;
    }
  }
  if (!data) data = await fetchNoteJson();
  noteData = data.note;
  transcriptData = data.transcript || [];
  provenanceData = data.provenance || {};
  flagsData = data.flags || {};
  baselineValues = {};
  pendingEdits = {};
  translatedValues = {};
  translatedTranscript = [];
  translationsByLang = {};
  verificationData = null;
  lastVerificationState = null;
  showCheckedTranscript = false;
  updateVerificationBanner(null);
  updateCheckedTranscriptToggle();

  // Latency fix: the note is already in hand (prefetched as soon as
  // l4_extract finished) — render it and switch screens IMMEDIATELY, in
  // source language. Never await a translation round trip before first
  // paint: a cold /translate call is an Ollama call and can take 30-120s,
  // which used to be the entire "drafting done" -> "note visible" gap.
  renderReview();
  renderDrawer();
  showScreen("review");

  // Bug 1: the selected language (default "en") may mismatch the note's
  // dominant script (e.g. a Hindi/Urdu-script consultation with English
  // still selected) — fire the translation in the BACKGROUND and re-render
  // in place once it lands, instead of blocking the screen switch on it.
  if (noteHasNonLatinScript(noteData, transcriptData)) {
    translateInBackgroundAndRerender();
  }
}

// Runs the initial auto-translate off the critical path of first paint.
// Reuses the exact same in-flight indicator and failure handling as the
// lang-select handler (runTranslationWithIndicator) — a "translating…" line
// on the way in, a silent fallback to source language plus the 4s
// "translation unavailable" message on failure. Re-renders in place once the
// translation lands (or fails), deferring if the reviewer is mid-edit — see
// isEditingActiveField / the deferredReviewRerenderPending check in
// onFieldBlur.
async function translateInBackgroundAndRerender() {
  langSelectEl.disabled = true;
  try {
    await runTranslationWithIndicator(refreshTranslations);
  } catch (err) {
    console.error("initial translation failed:", err);
  } finally {
    langSelectEl.disabled = false;
  }
  if (isEditingActiveField()) {
    deferredReviewRerenderPending = true;
    return;
  }
  renderReview();
  renderDrawer();
}

// True while the reviewer has an inline note-grid field focused for editing
// — used to defer a background re-render rather than clobber in-progress,
// unsaved keystrokes (medications/vitals cells use the same
// contenteditable .value convention as scalar/list fields).
function isEditingActiveField() {
  const el = document.activeElement;
  return !!(el && el.isContentEditable && noteGrid.contains(el));
}

// Fix 4 / Bug 1 (target-language-absolute): resolve what to *show* for a
// field — the translated overlay when one exists AND actually differs from
// the source (a pass-through entry — the source was already in the target
// language/script — is not "translated" for display purposes), else the raw
// value. "en" is a real target like hi/mr now, so this is not gated on
// currentLang at all: translatedValues is always the CURRENT language's
// overlay (populated by refreshTranslations(), including for "en").
// Medications/vitals paths never have a translatedValues entry (backend/mock
// never populate them), so they always fall through to the original value.
function hasTranslatedOverlay(path, rawValue) {
  const translated = translatedValues[path];
  return translated !== undefined && translated !== rawValue;
}

function displayedText(path, rawValue) {
  if (hasTranslatedOverlay(path, rawValue)) {
    return { text: translatedValues[path], isTranslated: true };
  }
  return { text: rawValue, isTranslated: false };
}

function valueCellHtml(path, rawValue) {
  const isEmpty = rawValue === null || rawValue === undefined || rawValue === "";
  baselineValues[path] = isEmpty ? "" : String(rawValue);
  const shown = isEmpty ? { text: null, isTranslated: false } : displayedText(path, rawValue);
  const display = isEmpty ? t("empty") : escapeHtml(shown.text);
  return `
    <div class="value" contenteditable="true" data-path="${path}" data-empty="${isEmpty}" data-translated="${shown.isTranslated}">${display}</div>
    <span class="editing-note" data-editing-note-for="${path}" hidden>(${u("editing_original")})</span>`;
}

function scalarFieldCell(labelKey, path) {
  const raw = getValueAtPath(noteData, path);
  const flagged = isFlagged(path);
  const hasDetail = provenanceData[path] || hasTranslatedOverlay(path, raw) || verificationDiffFor(path);

  return `
  <div class="cell field ${flagged ? "flagged" : ""}" data-path="${path}">
    <div class="label">
      <span>${t(labelKey)}</span>
      ${flagged ? `<span class="badge">${t("verify")}</span>` : ""}
      ${hasDetail ? `<button class="prov-trigger" data-prov-path="${path}">source</button>` : ""}
    </div>
    ${valueCellHtml(path, raw)}
    <div class="prov" data-prov-for="${path}" hidden></div>
  </div>`;
}

function itemRow(path, rawValue, detailText, leadingLabel) {
  const flagged = isFlagged(path);
  const hasDetail = provenanceData[path] || hasTranslatedOverlay(path, rawValue) || verificationDiffFor(path);

  return `
  <div class="item-row ${flagged ? "flagged" : ""}" data-path="${path}">
    ${leadingLabel ? `<span class="item-detail">${escapeHtml(leadingLabel)}</span>` : ""}
    ${hasDetail ? `<button class="prov-trigger" data-prov-path="${path}">source</button>` : ""}
    ${valueCellHtml(path, rawValue)}
    ${flagged ? `<span class="badge">${t("verify")}</span>` : ""}
    ${detailText ? `<span class="item-detail">${escapeHtml(detailText)}</span>` : ""}
  </div>
  <div class="prov" data-prov-for="${path}" hidden></div>`;
}

function listCell(labelKey, items, itemsHtml) {
  const body = items.length
    ? itemsHtml
    : `<div class="item-row"><div class="value" data-empty="true">${t("empty")}</div></div>`;
  // Bare-name flags (e.g. low_confidence_fields entry "vitals", resolved by
  // web/provenance.py's flags_by_path to the group's own field name) concern
  // the whole group, not one item[i] row — surface them here, since itemRow
  // only ever renders per-item paths like "vitals[0].value" (bug: this flag
  // was previously resolved server-side but never rendered anywhere).
  const flagReason = flagsData[labelKey];
  return `
  <div class="cell ${flagReason ? "flagged" : ""}" data-group="${labelKey}">
    <div class="label"><span>${t(labelKey)}</span>${flagReason ? `<span class="badge">${t("verify")}</span>` : ""}</div>
    ${body}
  </div>`;
}

function symptomsCellHtml() {
  const items = noteData.symptoms || [];
  const rows = items.map((s, i) => {
    const detailParts = [];
    if (s.finding_status && s.finding_status !== "Present") detailParts.push(s.finding_status);
    if (s.severity) detailParts.push(s.severity);
    if (s.since) detailParts.push(`since ${s.since}`);
    return itemRow(`symptoms[${i}].name`, s.name, detailParts.join(", "));
  }).join("");
  return listCell("symptoms", items, rows);
}

function vitalsCellHtml() {
  const items = noteData.vitals || [];
  // vitals[i].value is the only editable/provenance path; vital name is a
  // fixed, non-editable leading label (not part of the field-path notation).
  const rows = items.map((v, i) => itemRow(`vitals[${i}].value`, v.value, "", v.name)).join("");
  return listCell("vitals", items, rows);
}

function diagnosisCellHtml() {
  const items = noteData.diagnosis || [];
  const rows = items.map((d, i) => itemRow(`diagnosis[${i}].term`, d.term, d.status)).join("");
  return listCell("diagnosis", items, rows);
}

function stringListCellHtml(labelKey, fieldKey) {
  const items = noteData[fieldKey] || [];
  const rows = items.map((v, i) => itemRow(`${fieldKey}[${i}]`, v, "")).join("");
  return listCell(labelKey, items, rows);
}

function medicationsCellHtml() {
  const meds = noteData.medications || [];
  if (!meds.length) {
    return `<div class="cell"><div class="label"><span>${t("medications")}</span></div>
      <div class="value" data-empty="true">${t("empty")}</div></div>`;
  }
  const cols = ["drug", "dose", "frequency", "timing", "duration"];
  const header = cols.map((c) => `<th>${t(c)}</th>`).join("");
  const rows = meds.map((med, i) => {
    const cells = cols.map((c) => {
      const path = `medications[${i}].${c}`;
      const raw = med[c];
      const isEmpty = raw === null || raw === undefined || raw === "";
      const display = isEmpty ? t("empty") : escapeHtml(raw);
      baselineValues[path] = isEmpty ? "" : String(raw);
      const flagged = isFlagged(path);
      const hasProv = provenanceData[path] || verificationDiffFor(path);
      return `<td class="${flagged ? "flagged" : ""}" data-path="${path}">
        <div class="rx-cell-wrap">
          ${hasProv ? `<button class="prov-trigger" data-prov-path="${path}">source</button>` : ""}
          <div class="value" contenteditable="true" data-path="${path}" data-empty="${isEmpty}">${display}</div>
          ${flagged ? `<span class="badge">${t("verify")}</span>` : ""}
        </div>
        <div class="prov" data-prov-for="${path}" hidden></div>
      </td>`;
    }).join("");
    return `<tr>${cells}</tr>`;
  }).join("");
  return `
  <div class="cell">
    <div class="label"><span>${t("rx")}</span></div>
    <table class="rx-table"><thead><tr>${header}</tr></thead><tbody>${rows}</tbody></table>
  </div>`;
}

function renderReview() {
  document.getElementById("review-title").textContent = t("review_title");
  document.getElementById("view-transcript-btn").textContent = t("view_transcript");
  document.getElementById("label-patient").textContent = t("patient");
  document.getElementById("save-btn").textContent = u("save");
  document.getElementById("sign-btn").textContent = t("sign");
  document.getElementById("draft-pdf-link").textContent = u("draft_pdf");
  document.getElementById("draft-pdf-link").href = sessionId ? `/api/sessions/${sessionId}/pdf?lang=${currentLang}` : "#";
  document.getElementById("label-doctor-name").textContent = t("doctor_name");
  document.getElementById("label-reg-no").textContent = t("reg_no");
  document.getElementById("label-clinic").textContent = t("clinic_address");
  document.getElementById("sign-submit-btn").textContent = t("sign");

  noteGrid.innerHTML = [
    scalarFieldCell("chief_complaint", "chief_complaint"),
    scalarFieldCell("history", "history"),
    vitalsCellHtml(),
    symptomsCellHtml(),
    scalarFieldCell("examination", "examination"),
    diagnosisCellHtml(),
    medicationsCellHtml(),
    stringListCellHtml("investigations", "investigations"),
    stringListCellHtml("diagnostic_results", "diagnostic_results"),
    scalarFieldCell("advice", "advice"),
    scalarFieldCell("follow_up", "follow_up"),
  ].join("");

  const general = flagsData._general || [];
  if (general.length) {
    generalFlagsEl.hidden = false;
    generalFlagsEl.innerHTML = `<div class="label">${t("verify")}</div><ul>${
      general.map((r) => `<li>${escapeHtml(r)}</li>`).join("")
    }</ul>`;
  } else {
    generalFlagsEl.hidden = true;
  }

  attachFieldHandlers();
}

function attachFieldHandlers() {
  noteGrid.querySelectorAll(".value[contenteditable]").forEach((el) => {
    el.addEventListener("focus", () => {
      if (el.dataset.empty === "true") { el.textContent = ""; el.dataset.empty = "false"; }
      // Edits always target the source-language field: swap the translated
      // display back to the original the moment the box is focused, and say
      // so via the small label — machine translation is a display aid, the
      // edit of record is never the translation.
      if (el.dataset.translated === "true") {
        el.textContent = baselineValues[el.dataset.path] || "";
        const note = noteGrid.querySelector(`[data-editing-note-for="${el.dataset.path}"]`);
        if (note) note.hidden = false;
      }
    });
    el.addEventListener("blur", () => onFieldBlur(el));
  });

  noteGrid.querySelectorAll(".prov-trigger").forEach((btn) => {
    btn.addEventListener("click", () => toggleProvenance(btn.dataset.provPath));
  });
}

function onFieldBlur(el) {
  const path = el.dataset.path;
  const text = el.textContent.trim();
  const wasEmpty = text === "";
  const newVal = wasEmpty ? null : text;
  const oldVal = baselineValues[path] === "" ? null : baselineValues[path];
  if (newVal !== oldVal) {
    pendingEdits[path] = { field: path, old: oldVal, new: newVal };
  } else {
    delete pendingEdits[path];
  }

  const note = noteGrid.querySelector(`[data-editing-note-for="${path}"]`);
  if (note) note.hidden = true;

  if (wasEmpty) {
    el.dataset.empty = "true";
    el.textContent = t("empty");
  } else if (newVal === oldVal && hasTranslatedOverlay(path, baselineValues[path])) {
    // Unchanged and a translation exists for this language: editing is
    // done, resume showing the translated overlay (the original was only
    // shown transiently while the field had focus).
    el.textContent = translatedValues[path];
  }

  // A background translation may have landed while this field (or another
  // one — focus can move directly between fields on the same blur/focus
  // tick) was being edited. Apply it now, but only once nothing in the note
  // grid is still being edited. Checked unconditionally (not skipped for an
  // emptied field) so the deferred re-render is never lost.
  if (deferredReviewRerenderPending && !isEditingActiveField()) {
    deferredReviewRerenderPending = false;
    renderReview();
    renderDrawer();
  }
}

function toggleProvenance(path) {
  const panel = noteGrid.querySelector(`.prov[data-prov-for="${path}"]`);
  if (!panel) return;
  const isOpen = !panel.hidden;
  const drawerOpen = drawerEl.classList.contains("open");
  if (isOpen && drawerOpen) {
    panel.hidden = true;
    return;
  }
  if (isOpen && !drawerOpen) {
    // Defensive desync recovery: the panel's own flag says open but the
    // drawer visually is not (e.g. a future path closes the drawer without
    // going through closeDrawer()) — re-open + re-scroll rather than
    // silently closing a panel the user can no longer see, which would
    // read as the button doing nothing.
    const prov = provenanceData[path];
    if (prov) scrollDrawerToTurn(prov.turn_index);
    return;
  }
  const parts = [];
  if (hasTranslatedOverlay(path, baselineValues[path])) {
    parts.push(
      `<div class="original-line">${u("original")}: ${escapeHtml(baselineValues[path] || "")}</div>`
    );
  }
  const prov = provenanceData[path];
  if (prov) {
    parts.push(
      `<div class="source-line">${escapeHtml(prov.snippet)} ${formatTs(prov.start, prov.end)}</div>`
    );
  }
  const diff = verificationDiffFor(path);
  if (diff) {
    const diffText = `fast: ${diff.fast_value ?? ""} / check: ${diff.accurate_value ?? ""}`;
    parts.push(
      `<div class="verify-diff"><span class="verify-diff-text">${escapeHtml(diffText)}</span>` +
        `<button class="use-checked-btn" data-use-checked-path="${escapeHtml(path)}">${u("use_checked_version")}</button></div>`
    );
  }
  if (!parts.length) return;
  panel.innerHTML = parts.join("");
  panel.hidden = false;
  if (prov) scrollDrawerToTurn(prov.turn_index);
}

// Event delegation: the "use checked version" button lives inside a .prov
// panel that is rendered on demand (toggleProvenance), long after
// attachFieldHandlers() ran — attached once on the stable noteGrid container
// rather than per-render, since noteGrid.innerHTML is replaced wholesale on
// every renderReview() call but noteGrid itself never is.
noteGrid.addEventListener("click", (e) => {
  const btn = e.target.closest(".use-checked-btn");
  if (!btn) return;
  applyCheckedVersion(btn.dataset.useCheckedPath).catch((err) =>
    console.error("use checked version failed:", err)
  );
});

/* ---------- Save (PATCH) ---------- */

document.getElementById("save-btn").addEventListener("click", async () => {
  const edits = Object.values(pendingEdits);
  if (!edits.length) return;
  const res = await api(`/api/sessions/${sessionId}/note`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ edits }),
  });
  const data = await res.json();
  noteData = data.note;
  for (const edit of edits) baselineValues[edit.field] = edit.new ?? "";
  pendingEdits = {};
});

/* ---------- Draft PDF (print, staying on the review page) ----------
 * "DRAFT PDF AND PRINT" must overlay the browser's print dialog on the
 * CURRENT review page — no new tab, no focus jump. The /pdf endpoint always
 * answers with Content-Disposition: attachment (web/app.py get_pdf) so
 * pointing a plain iframe.src straight at that URL only forces a file
 * download — there is never an inline, printable document to call .print()
 * on. Fetching the bytes ourselves and handing the browser a blob: URL
 * (which carries no Content-Disposition) sidesteps that and renders inline.
 * Printing is done from a hidden iframe's own 'load' event: verified against
 * headless Chrome that Chromium's built-in PDF viewer never initializes
 * (and 'load' never fires) inside a zero-area frame, hence the real
 * off-screen size below, not 0x0. A new tab is opened ONLY as a fallback, if
 * the iframe's print() itself throws (or the fetch/blob step fails) — never
 * as a matter of course.
 */
document.getElementById("draft-pdf-link").addEventListener("click", async (event) => {
  if (!sessionId) return;
  event.preventDefault();
  const url = event.currentTarget.href;
  try {
    const res = await api(url);
    if (!res.ok) throw new Error(`pdf fetch failed: ${res.status}`);
    const blobUrl = URL.createObjectURL(await res.blob());
    printBlobViaHiddenIframe(blobUrl, url);
  } catch (err) {
    console.error("draft pdf fetch failed, falling back to opening a new tab:", err);
    window.open(url, "_blank");
  }
});

function printBlobViaHiddenIframe(blobUrl, fallbackUrl) {
  const iframe = document.createElement("iframe");
  iframe.style.cssText = "position:fixed; left:-9999px; top:-9999px; width:600px; height:800px; border:0;";
  iframe.setAttribute("aria-hidden", "true");
  iframe.addEventListener("load", () => {
    try {
      iframe.contentWindow.print();
    } catch (err) {
      console.error("draft pdf print() failed, falling back to opening a new tab:", err);
      window.open(fallbackUrl, "_blank");
    }
    setTimeout(() => {
      iframe.remove();
      URL.revokeObjectURL(blobUrl);
    }, 2000);
  });
  iframe.src = blobUrl;
  document.body.appendChild(iframe);
}

/* ---------- Transcript drawer ---------- */

function renderDrawer() {
  updateCheckedTranscriptToggle();
  if (showCheckedTranscript && verificationData) {
    drawerTurnsEl.innerHTML = (verificationData.transcript_accurate || []).map((w) => `
      <div class="turn">
        <span class="role">${escapeHtml(u("checked_transcript"))}</span>
        ${escapeHtml(w.text)}
        <span class="ts">${formatTs(w.start, w.end)}</span>
      </div>
    `).join("");
    return;
  }
  drawerTurnsEl.innerHTML = transcriptData.map((turn, i) => {
    const overlay = translatedTranscript[i];
    const translated = overlay !== undefined && overlay !== turn.text ? overlay : undefined;
    return `
    <div class="turn" data-turn-index="${i}">
      <span class="role">${escapeHtml(turn.speaker_role)}</span>
      ${escapeHtml(translated !== undefined ? translated : turn.text)}
      ${translated !== undefined ? `<div class="turn-original">${u("original")}: ${escapeHtml(turn.text)}</div>` : ""}
      <span class="ts">${formatTs(turn.start, turn.end)}</span>
    </div>
  `;
  }).join("");
}

checkedTranscriptToggleEl.addEventListener("click", () => {
  showCheckedTranscript = !showCheckedTranscript;
  renderDrawer();
});

function scrollDrawerToTurn(turnIndex) {
  drawerEl.classList.add("open");
  const turnEl = drawerTurnsEl.querySelector(`[data-turn-index="${turnIndex}"]`);
  if (!turnEl) return;
  turnEl.scrollIntoView({ block: "center", behavior: "smooth" });
  turnEl.classList.add("highlight");
  requestAnimationFrame(() => {
    setTimeout(() => turnEl.classList.remove("highlight"), 50);
  });
}

document.getElementById("view-transcript-btn").addEventListener("click", () => {
  drawerEl.classList.toggle("open");
});

// Bug 2: the drawer must never trap the user — close via the pinned CLOSE
// button, Escape, or a click anywhere outside it.
// Bug A (drawer/panel state desync): closing the drawer must also reset
// every per-field .prov panel's own open/closed flag, so the two states
// (drawer open, panel hidden) never fall out of sync — otherwise re-clicking
// the same source button after a close reads panel.hidden as still "open"
// and toggleProvenance() silently closes it instead of reopening the drawer.
function closeDrawer() {
  drawerEl.classList.remove("open");
  document.querySelectorAll(".prov").forEach((panel) => {
    panel.hidden = true;
  });
}

document.getElementById("drawer-close-btn").addEventListener("click", closeDrawer);

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && drawerEl.classList.contains("open")) closeDrawer();
});

document.addEventListener("click", (e) => {
  if (!drawerEl.classList.contains("open")) return;
  if (drawerEl.contains(e.target)) return;
  // Elements that legitimately OPEN the drawer as part of this same click
  // must not immediately re-close it via this outside-click handler.
  if (e.target.closest("#view-transcript-btn, .prov-trigger")) return;
  closeDrawer();
});

/* ---------- Language switching ---------- */

// A cold /translate call (uncached lang, and/or a loaded Ollama host) can
// take 1-2 minutes with nothing else on screen changing meanwhile — the
// doctor gets no feedback that anything is happening. Delayed-show guard:
// the "translating…" line only appears if the request is STILL in flight
// after TRANSLATE_INDICATOR_DELAY_MS, so an already-cached repeat switch
// (which resolves in a microtask, well under the delay) never flashes it.
const TRANSLATE_INDICATOR_DELAY_MS = 150;
const TRANSLATE_ERROR_DISPLAY_MS = 4000;
const translateStatusEl = document.getElementById("translate-status");
const langSelectEl = document.getElementById("lang-select");
let translateIndicatorTimer = null;
let translateErrorTimer = null;

function hideTranslateStatus() {
  clearTimeout(translateErrorTimer);
  translateErrorTimer = null;
  translateStatusEl.hidden = true;
  translateStatusEl.classList.remove("error");
}

// Shared in-flight indicator machinery: shows "translating…" once a request
// has genuinely been in flight for TRANSLATE_INDICATOR_DELAY_MS (so an
// already-cached repeat, which resolves in a microtask, never flashes it),
// and on failure shows "translation unavailable" for TRANSLATE_ERROR_DISPLAY_MS
// before falling silently back to source language. Used by both the
// lang-select handler AND the initial background auto-translate
// (translateInBackgroundAndRerender) so the doctor sees the same feedback
// regardless of which path triggered the translation. Rethrows on failure so
// each caller can log its own context-specific error message.
async function runTranslationWithIndicator(translateFn) {
  clearTimeout(translateErrorTimer);
  translateIndicatorTimer = setTimeout(() => {
    translateStatusEl.textContent = u("translating");
    translateStatusEl.classList.remove("error");
    translateStatusEl.hidden = false;
  }, TRANSLATE_INDICATOR_DELAY_MS);
  try {
    await translateFn();
    clearTimeout(translateIndicatorTimer);
    translateStatusEl.hidden = true;
  } catch (err) {
    clearTimeout(translateIndicatorTimer);
    translateStatusEl.textContent = u("translation_unavailable");
    translateStatusEl.classList.add("error");
    translateStatusEl.hidden = false;
    translateErrorTimer = setTimeout(hideTranslateStatus, TRANSLATE_ERROR_DISPLAY_MS);
    throw err;
  }
}

langSelectEl.addEventListener("change", async (e) => {
  currentLang = e.target.value;
  // The indicator is gated on an ACTUAL request being in flight (>150ms),
  // not on the target language: English is usually a client-side no-op
  // (cached / Latin source → instant, so the delayed-show guard means no
  // flash), but for a Devanagari/Arabic-script source note English is a
  // real backend translation too — a silent 1-2 min wait there is exactly
  // the gap this indicator exists to close. Cached per lang server-side
  // and client-side, so repeated switching never re-hits Ollama for a
  // language already fetched this session.
  langSelectEl.disabled = true;
  try {
    await runTranslationWithIndicator(refreshTranslations);
  } catch (err) {
    console.error("translation fetch failed:", err);
  } finally {
    langSelectEl.disabled = false;
  }
  renderReview();
  renderDrawer();
});

async function refreshTranslations() {
  if (translationsByLang[currentLang]) {
    ({ note_values: translatedValues, transcript: translatedTranscript } = translationsByLang[currentLang]);
    return;
  }
  try {
    const res = await api(`/api/sessions/${sessionId}/translate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lang: currentLang }),
    });
    if (!res.ok) throw new Error(`translate failed: ${res.status}`);
    const data = await res.json();
    translationsByLang[currentLang] = data;
    translatedValues = data.note_values || {};
    translatedTranscript = data.transcript || [];
  } catch (err) {
    // Failure (network or non-200): fall back to the source-language values
    // — never leave a stale overlay from a previously selected language on
    // screen mislabeled as the newly selected one.
    translatedValues = {};
    translatedTranscript = [];
    throw err;
  }
}

/* ---------- Sign flow ---------- */

const signBtn = document.getElementById("sign-btn");
const signForm = document.getElementById("sign-form");
const signedPanel = document.getElementById("signed-panel");

// Bug 3: the review page (note-grid + footer) commonly runs taller than the
// browser viewport, so toggling sign-form/signed-panel visible produced no
// change *within the user's current view* — indistinguishable from "the
// button did nothing". Scroll the newly-shown element into view every time.
signBtn.addEventListener("click", () => {
  signForm.hidden = !signForm.hidden;
  document.getElementById("sign-doctor-name").value = localStorage.getItem("cliniscribe_doctor_name") || "";
  document.getElementById("sign-reg-no").value = localStorage.getItem("cliniscribe_reg_no") || "";
  document.getElementById("sign-clinic").value = localStorage.getItem("cliniscribe_clinic") || "";
  if (!signForm.hidden) signForm.scrollIntoView({ block: "center", behavior: "smooth" });
});

document.getElementById("sign-submit-btn").addEventListener("click", async () => {
  const name = document.getElementById("sign-doctor-name").value.trim();
  const regNo = document.getElementById("sign-reg-no").value.trim();
  const clinic = document.getElementById("sign-clinic").value.trim();

  localStorage.setItem("cliniscribe_doctor_name", name);
  localStorage.setItem("cliniscribe_reg_no", regNo);
  localStorage.setItem("cliniscribe_clinic", clinic);

  let res;
  try {
    res = await api(`/api/sessions/${sessionId}/sign`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lang: currentLang, doctor: { name, reg_no: regNo, clinic } }),
    });
  } catch (err) {
    console.error("sign request failed:", err);
    return;
  }
  if (!res.ok) {
    // Never silently claim "signed" on a server-side failure — leave the
    // form open so the physician can see something went wrong and retry.
    console.error("sign request returned", res.status);
    return;
  }
  const data = await res.json();

  signForm.hidden = true;
  signBtn.hidden = true;
  signedPanel.hidden = false;
  // Sign interplay: note.json is now final — re-render the verification
  // banner immediately in its read-only, post-sign wording (if a result
  // already exists) rather than waiting for the next 1s poll tick, and drop
  // any live per-field flags/edit affordance verificationDiffFor() gates on
  // isSignedNow().
  updateVerificationBanner(lastVerificationStatusObj);
  renderReview();
  document.getElementById("signed-label").textContent = t("signed");
  const link = document.getElementById("signed-pdf-link");
  link.textContent = u("download_pdf");
  link.href = data.pdf_url || `/api/sessions/${sessionId}/pdf?lang=${currentLang}`;
  signedPanel.scrollIntoView({ block: "center", behavior: "smooth" });
});

/* =====================================================================
 * Init
 * ===================================================================== */

async function loadTranslations() {
  const res = await api("/api/translations");
  translations = await res.json();
}

async function init() {
  // Demo banner: only when opened as a file (window.MOCK), never over http —
  // fixes users mistaking mock fixture values (Cough, BP 120/80, ...) for
  // pipeline hallucinations. Permanent (not toggled by screen) on both screens.
  document.getElementById("demo-banner").hidden = !window.MOCK;

  await loadTranslations();
  document.title = t("app_title");
  captureCaption.textContent = t("capture_hint");
  micRing.setAttribute("aria-label", t("record"));
  fileHintText.textContent = u("use_audio_file");
  livePreviewLabelEl.textContent = u("live_preview_label");
  stageLinesEl.setAttribute("aria-label", t("processing"));
  document.getElementById("lang-select").setAttribute("aria-label", t("language"));
  const stageKeyToText = { transcribed: "stage_transcribed", speakers: "stage_speakers", drafting: "stage_drafting" };
  stageLinesEl.querySelectorAll(".stage-line").forEach((el) => {
    el.querySelector(".text").textContent = t(stageKeyToText[el.dataset.stage]);
  });

  // Session-restore-by-hash: reopen an already-processed session's review
  // screen directly (e.g. #session=20260710-213933-8c4119) instead of
  // re-recording. The SPA has no other session-restore path, and reloading
  // is otherwise a dead end once a session has moved past the capture screen.
  const sessionMatch = location.hash.match(/^#session=([A-Za-z0-9_-]+)$/);
  if (sessionMatch) {
    sessionId = sessionMatch[1];
    await loadNoteAndShowReview();
    // One-shot verification check (no live poll on this restore path, same
    // as every other field on a restored session — see (d)'s "same status
    // poll" note, which only applies to the live capture->review flow).
    try {
      const res = await api(`/api/sessions/${sessionId}/status`);
      if (res.ok) {
        const status = await res.json();
        if (status.verification) {
          handleVerificationStatus(status.verification);
          if (status.verification.state === "done" || status.verification.state === "failed") {
            await fetchVerificationDetails();
          }
        }
      }
    } catch (err) {
      console.error("verification status check failed:", err);
    }
    return;
  }
  showScreen("capture");
}

init();

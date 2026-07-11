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
  draft_pdf: "draft pdf",
  download_pdf: "download pdf",
  original: "original",
  editing_original: "editing original",
  live_preview_label: "LIVE PREVIEW — final transcript follows",
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

// Real transcription %/ETA suffix on the "transcribed" line (spec: appears
// only once the first progress datum arrives, disappears when l3_asr ends —
// status.json's "progress"/"eta_seconds" are absent outside that window).
function formatEtaSuffix(etaSeconds) {
  if (typeof etaSeconds !== "number") return "";
  const rounded = etaSeconds < 60
    ? `~${Math.round(etaSeconds)}s left`
    : `~${Math.round(etaSeconds / 60)} min left`;
  return ` · ${rounded}`;
}

function updateTranscribeProgress(status) {
  const el = stageLinesEl.querySelector('[data-stage="transcribed"] .progress');
  if (status.stage !== "l3_asr" || typeof status.progress !== "number") {
    el.hidden = true;
    el.textContent = "";
    return;
  }
  const pct = Math.round(status.progress * 100);
  el.textContent = ` · ${pct}%${formatEtaSuffix(status.eta_seconds)}`;
  el.hidden = false;
}

function applyStatus(status) {
  const done = new Set(status.stages_done || []);
  const stage = status.stage;

  const transcribedDone = done.has("l1_preprocess") && done.has("l2_diarize") && done.has("l3_asr");
  setStageLine("transcribed", transcribedDone ? "done" : (stage && ["l1_preprocess", "l2_diarize", "l3_asr"].includes(stage) ? "active" : ""));
  updateTranscribeProgress(status);

  const speakersDone = done.has("l3_5_normalize");
  setStageLine("speakers", speakersDone ? "done" : (stage === "l3_5_normalize" ? "active" : ""));

  const draftingDone = done.has("l4_extract");
  setStageLine("drafting", draftingDone ? "done" : (stage === "l4_extract" ? "active" : ""));
}

function startStatusPolling() {
  applyStatus({ state: "processing", stage: null, stages_done: [] });
  statusPollHandle = setInterval(async () => {
    // A transient 500 (e.g. a status.json write in progress server-side)
    // must never break polling — swallow and silently retry next tick.
    try {
      const res = await api(`/api/sessions/${sessionId}/status`);
      if (!res.ok) return;
      const status = await res.json();
      applyStatus(status);
      if (status.state === "review") {
        clearInterval(statusPollHandle);
        await loadNoteAndShowReview();
      }
      // status.error is surfaced only via console — no dashboard element per spec.
      if (status.error) console.error("pipeline error:", status.error);
    } catch (err) {
      console.error("status poll failed, retrying next tick:", err);
    }
  }, 1000);
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

async function loadNoteAndShowReview() {
  const res = await api(`/api/sessions/${sessionId}/note`);
  const data = await res.json();
  noteData = data.note;
  transcriptData = data.transcript || [];
  provenanceData = data.provenance || {};
  flagsData = data.flags || {};
  baselineValues = {};
  pendingEdits = {};
  translatedValues = {};
  translatedTranscript = [];
  translationsByLang = {};
  // Bug 1: the selected language (default "en") may mismatch the note's
  // dominant script (e.g. a Hindi/Urdu-script consultation with English
  // still selected) — fire the translation immediately so the first paint
  // is already in the selected language, not a tap-to-translate surprise.
  if (noteHasNonLatinScript(noteData, transcriptData)) {
    await refreshTranslations();
  }
  renderReview();
  renderDrawer();
  showScreen("review");
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
  const flagReason = flagsData[path];
  const hasDetail = provenanceData[path] || hasTranslatedOverlay(path, raw);

  return `
  <div class="cell field ${flagReason ? "flagged" : ""}" data-path="${path}">
    <div class="label">
      <span>${t(labelKey)}</span>
      ${flagReason ? `<span class="badge">${t("verify")}</span>` : ""}
      ${hasDetail ? `<button class="prov-trigger" data-prov-path="${path}">source</button>` : ""}
    </div>
    ${valueCellHtml(path, raw)}
    <div class="prov" data-prov-for="${path}" hidden></div>
  </div>`;
}

function itemRow(path, rawValue, detailText, leadingLabel) {
  const flagReason = flagsData[path];
  const hasDetail = provenanceData[path] || hasTranslatedOverlay(path, rawValue);

  return `
  <div class="item-row ${flagReason ? "flagged" : ""}" data-path="${path}">
    ${leadingLabel ? `<span class="item-detail">${escapeHtml(leadingLabel)}</span>` : ""}
    ${hasDetail ? `<button class="prov-trigger" data-prov-path="${path}">source</button>` : ""}
    ${valueCellHtml(path, rawValue)}
    ${flagReason ? `<span class="badge">${t("verify")}</span>` : ""}
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
      const flagReason = flagsData[path];
      const hasProv = provenanceData[path];
      return `<td class="${flagReason ? "flagged" : ""}" data-path="${path}">
        <div class="rx-cell-wrap">
          ${hasProv ? `<button class="prov-trigger" data-prov-path="${path}">source</button>` : ""}
          <div class="value" contenteditable="true" data-path="${path}" data-empty="${isEmpty}">${display}</div>
          ${flagReason ? `<span class="badge">${t("verify")}</span>` : ""}
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
    return;
  }
  // Unchanged and a translation exists for this language: editing is done,
  // resume showing the translated overlay (the original was only shown
  // transiently while the field had focus).
  if (newVal === oldVal && hasTranslatedOverlay(path, baselineValues[path])) {
    el.textContent = translatedValues[path];
  }
}

function toggleProvenance(path) {
  const panel = noteGrid.querySelector(`.prov[data-prov-for="${path}"]`);
  if (!panel) return;
  const isOpen = !panel.hidden;
  if (isOpen) {
    panel.hidden = true;
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
  if (!parts.length) return;
  panel.innerHTML = parts.join("");
  panel.hidden = false;
  if (prov) scrollDrawerToTurn(prov.turn_index);
}

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

/* ---------- Transcript drawer ---------- */

function renderDrawer() {
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
function closeDrawer() {
  drawerEl.classList.remove("open");
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

document.getElementById("lang-select").addEventListener("change", async (e) => {
  currentLang = e.target.value;
  // Bug 1 (target-language-absolute): "en" is a real target like hi/mr —
  // a Devanagari/Arabic-script note still needs translating TO English, so
  // every switch (including back to en) refreshes the overlay. Cached per
  // lang server-side and client-side, so repeated en<->hi<->mr switching
  // never re-hits Ollama for a language already fetched this session.
  await refreshTranslations();
  renderReview();
  renderDrawer();
});

async function refreshTranslations() {
  if (translationsByLang[currentLang]) {
    ({ note_values: translatedValues, transcript: translatedTranscript } = translationsByLang[currentLang]);
    return;
  }
  const res = await api(`/api/sessions/${sessionId}/translate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lang: currentLang }),
  });
  const data = await res.json();
  translationsByLang[currentLang] = data;
  translatedValues = data.note_values || {};
  translatedTranscript = data.transcript || [];
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
    return;
  }
  showScreen("capture");
}

init();

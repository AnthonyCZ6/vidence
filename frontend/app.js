// Vidence — interfaz de demostración. Sin dependencias ni recursos externos.
// Todo texto que proviene de la evidencia se inserta como texto (nunca como HTML).

const API = "/api";
const POLL_INTERVAL_MS = 700;
const RECORDER_TIMESLICE_MS = 1000;
const UNCERTAIN_WORD_PROBABILITY = 0.5;

const STAGES = [
  ["upload", "Recepción, SHA-256 y cifrado AES-256"],
  ["verifying", "Verificación de integridad del original"],
  ["probing", "Metadatos (ffprobe)"],
  ["enhancing", "Mejora de audio y video (RNNoise + normalización)"],
  ["transcribing", "Transcripción con Whisper (CPU int8)"],
  ["frames", "Análisis de video y fotogramas con IA (YOLOv8)"],
  ["finalizing", "Cifrado y resguardo de resultados"],
];
const STATUS_LABELS = { processed: "Procesada", processing: "Procesando", failed: "Error", pending: "Pendiente" };
const SOURCE_LABELS = { recording: "Grabación en el navegador", upload: "Archivo subido" };
const ROLE_LABELS = { original: "Original", original_preview: "Vista previa sin filtros", enhanced: "Mejorado" };
const RECORDER_TYPES = {
  audio: ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"],
  video: ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm", "video/mp4"],
};

const state = { selected: null, pollTimer: null, seenStages: new Set() };
const recorder = { stream: null, media: null, chunks: [], kind: "audio", startedAt: 0, timer: null };

const $ = (id) => document.getElementById(id);

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child !== null && child !== undefined && child !== false) node.append(child);
  }
  return node;
}

// --- Formato --------------------------------------------------------------------

function formatClock(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  const total = Math.max(0, Math.floor(seconds));
  const pad = (n) => String(n).padStart(2, "0");
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  return hours ? `${hours}:${pad(minutes)}:${pad(total % 60)}` : `${pad(minutes)}:${pad(total % 60)}`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}

const formatDate = (iso) => (iso ? new Date(iso).toLocaleString("es") : "—");
const shortHash = (hash) => `${hash.slice(0, 12)}…${hash.slice(-6)}`;
const assetUrl = (evidenceId, name) =>
  `${API}/evidence/${evidenceId}/assets/${name.split("/").map(encodeURIComponent).join("/")}`;

// --- API ------------------------------------------------------------------------

async function describeError(response) {
  try {
    const body = await response.json();
    if (Array.isArray(body.detail)) return body.detail.map((d) => d.msg).join("; ");
    return body.detail || `Error ${response.status}`;
  } catch {
    return `Error ${response.status}`;
  }
}

async function getJSON(path) {
  const response = await fetch(`${API}${path}`, { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(await describeError(response));
  return response.json();
}

function uploadEvidence(blob, filename, source, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API}/evidence?${new URLSearchParams({ filename, source })}`);
    xhr.setRequestHeader("X-Requested-With", "vidence");
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (event) => event.lengthComputable && onProgress(event.loaded / event.total);
    xhr.onload = () => {
      let body = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        body = null;
      }
      if (xhr.status === 201) resolve(body);
      else reject(new Error((body && body.detail) || `Error ${xhr.status}`));
    };
    xhr.onerror = () => reject(new Error("No se pudo contactar al servidor local"));
    xhr.send(blob);
  });
}

// --- Estado del sistema -------------------------------------------------------------

async function loadHealth() {
  const pills = $("status-pills");
  try {
    const health = await getJSON("/health");
    const visionName = (health.vision_model || "yolov8n").replace(/\.[^.]+$/, "");
    const items = [
      [health.offline_guard ? "ok" : "warn", health.offline_guard ? "Offline · red bloqueada" : "Guardia offline inactivo"],
      [health.whisper_ready ? "ok" : "warn", `Whisper ${health.whisper_model}${health.whisper_ready ? "" : " (falta modelo)"}`],
      [health.vision_ready ? "ok" : "warn", `Visión ${visionName}${health.vision_ready ? "" : " (falta modelo)"}`],
      ["info", health.compute],
      ["info", health.denoiser === "rnnoise" ? "RNNoise" : "Denoise FFT"],
    ];
    pills.replaceChildren(...items.map(([tone, text]) => el("li", { class: `pill pill-${tone}` }, text)));
  } catch {
    pills.replaceChildren(el("li", { class: "pill pill-warn" }, "Servidor no disponible"));
  }
}

// --- Lista de evidencias ----------------------------------------------------------------

async function loadList() {
  const items = await getJSON("/evidence");
  const rows = items.map((item) => {
    const kind = item.has_video ? "Video" : item.has_audio ? "Audio" : "—";
    return el("li", {},
      el("button", {
        type: "button",
        class: `evidence-item${item.evidence_id === state.selected ? " is-selected" : ""}`,
        "aria-current": item.evidence_id === state.selected ? "true" : null,
        onclick: () => selectEvidence(item.evidence_id),
      },
      el("span", { class: "evidence-name" }, item.original_filename),
      el("span", { class: "evidence-meta" }, `${kind} · ${formatClock(item.duration_s)} · ${formatDate(item.received_at)}`),
      el("span", { class: `badge badge-${item.status}` }, STATUS_LABELS[item.status])));
  });
  $("evidence-list").replaceChildren(...rows);
  $("list-empty").hidden = items.length > 0;
  return items;
}

// --- Progreso -------------------------------------------------------------------------------

function showProgress(filename) {
  clearTimeout(state.pollTimer);
  state.seenStages = new Set(["upload"]);
  $("progress").hidden = false;
  $("progress-file").textContent = filename;
  $("progress-error").hidden = true;
  $("stages").replaceChildren(...STAGES.map(([key, label]) =>
    el("li", { dataset: { stage: key } }, el("span", { class: "stage-label" }, label), el("span", { class: "stage-note" }))));
  renderStages("upload", 0, false);
}

function renderStages(activeKey, progress, finished) {
  const activeIndex = STAGES.findIndex(([key]) => key === activeKey);
  STAGES.forEach(([key], index) => {
    const item = $("stages").querySelector(`[data-stage="${key}"]`);
    const seen = state.seenStages.has(key);
    if (finished || index < activeIndex) item.className = seen ? "is-done" : "is-skipped";
    else item.className = index === activeIndex ? "is-active" : "";
  });
  const overall = finished ? 1 : (Math.max(activeIndex, 0) + (progress ?? 0.5)) / STAGES.length;
  $("bar-fill").style.width = `${Math.round(overall * 100)}%`;
}

function setStageNote(key, text) {
  const note = $("stages").querySelector(`[data-stage="${key}"] .stage-note`);
  if (note) note.textContent = text;
}

function failProgress(message) {
  const error = $("progress-error");
  error.textContent = message;
  error.hidden = false;
  $("stages").querySelector(".is-active")?.classList.replace("is-active", "is-failed");
}

function pollJob(jobId, evidenceId) {
  const tick = async () => {
    try {
      const job = await getJSON(`/jobs/${jobId}`);
      (job.stages_seen || []).forEach((stage) => state.seenStages.add(stage));
      if (job.stage === "done") {
        renderStages(null, 1, true);
        await loadList();
        await selectEvidence(evidenceId);
        return;
      }
      if (job.stage === "failed") {
        failProgress(job.error || "Error al procesar la evidencia");
        await loadList();
        return;
      }
      renderStages(job.stage === "queued" ? "verifying" : job.stage, job.progress, false);
      if (job.progress !== null) setStageNote(job.stage, `${Math.round(job.progress * 100)} %`);
      state.pollTimer = setTimeout(tick, POLL_INTERVAL_MS);
    } catch (error) {
      failProgress(error.message);
    }
  };
  tick();
}

async function submitEvidence(blob, filename, source) {
  showProgress(filename);
  try {
    const upload = await uploadEvidence(blob, filename, source, (fraction) => {
      setStageNote("upload", `${Math.round(fraction * 100)} %`);
      renderStages("upload", fraction, false);
    });
    setStageNote("upload", `SHA-256 ${shortHash(upload.sha256)}`);
    state.selected = upload.evidence_id;
    await loadList();
    pollJob(upload.job_id, upload.evidence_id);
  } catch (error) {
    failProgress(error.message);
  }
}

// --- Resultado ---------------------------------------------------------------------------------

async function selectEvidence(evidenceId) {
  state.selected = evidenceId;
  document.querySelectorAll(".evidence-item").forEach((button) => button.classList.remove("is-selected"));
  const result = $("result");
  result.hidden = false;
  try {
    const manifest = await getJSON(`/evidence/${evidenceId}`);
    result.replaceChildren(...renderResult(manifest));
  } catch (error) {
    result.replaceChildren(el("section", { class: "card" }, el("p", { class: "error" }, error.message)));
  }
  await loadList();
}

function renderResult(manifest) {
  const id = manifest.evidence_id;
  const byRole = Object.fromEntries(manifest.derivatives.map((asset) => [asset.role, asset]));
  const tag = manifest.media.has_video ? "video" : "audio";
  const makePlayer = (asset) => el(tag, { src: assetUrl(id, asset.name), controls: true, preload: "metadata", playsinline: true });
  const original = makePlayer(byRole.original_preview);
  const enhanced = makePlayer(byRole.enhanced);
  const seek = (seconds) => {
    enhanced.currentTime = seconds;
    enhanced.play();
    enhanced.scrollIntoView({ behavior: "smooth", block: "center" });
  };
  return [
    renderSummary(manifest),
    renderComparison(original, enhanced, byRole, manifest.media.has_video),
    renderTranscript(manifest.transcript, enhanced, seek),
    manifest.frames.length ? renderFrames(manifest, seek) : null,
    renderMetadata(manifest),
    renderLineage(manifest),
  ].filter(Boolean);
}

function renderSummary(manifest) {
  const id = manifest.evidence_id;
  return el("section", { class: "card summary" },
    el("div", {},
      el("h2", { class: "truncate" }, manifest.ingest.original_filename),
      el("p", { class: "muted" },
        `${SOURCE_LABELS[manifest.ingest.source]} · recibido ${formatDate(manifest.ingest.received_at)}`)),
    el("div", { class: "actions" },
      el("a", { class: "btn-secondary", href: `${API}/evidence/${id}/report`, download: "" }, "Reporte JSON"),
      el("a", { class: "btn-secondary", href: `${assetUrl(id, "original.bin")}?download=true` }, "Original")));
}

function renderComparison(original, enhanced, byRole, hasVideo) {
  const swap = () => {
    const [from, to] = original.paused ? [enhanced, original] : [original, enhanced];
    to.currentTime = from.currentTime;
    from.pause();
    to.play();
  };
  original.addEventListener("play", () => enhanced.pause());
  enhanced.addEventListener("play", () => original.pause());
  const column = (title, subtitle, player, asset) => el("div", { class: "player" },
    el("h3", {}, title), el("p", { class: "muted small" }, subtitle), player,
    el("p", { class: "hash small", title: asset.sha256 }, `SHA-256 ${shortHash(asset.sha256)}`));
  return el("section", { class: "card" },
    el("div", { class: "card-head" },
      el("h2", {}, "Antes y después"),
      el("button", { type: "button", class: "btn-secondary", onclick: swap }, "Alternar A/B en el mismo instante")),
    el("div", { class: "players" },
      column("Original", "Mismo códec, sin filtros", original, byRole.original_preview),
      column("Mejorado por IA local",
        hasVideo ? "RNNoise · EBU R128 · reducción de ruido y nitidez de imagen" : "RNNoise · normalización EBU R128",
        enhanced, byRole.enhanced)));
}

function renderTranscript(transcript, player, seek) {
  if (!transcript) {
    return el("section", { class: "card" }, el("h2", {}, "Transcripción"),
      el("p", { class: "muted" }, "El archivo no contiene audio."));
  }
  const items = transcript.segments.map((segment) => el("li", { class: "segment", dataset: { start: segment.start, end: segment.end } },
    el("button", { type: "button", class: "timestamp", onclick: () => seek(segment.start), title: "Reproducir desde aquí" },
      formatClock(segment.start)),
    el("p", {}, segment.words.length ? segment.words.map(renderWord) : segment.text)));
  const list = el("ol", { class: "segments" }, items.length ? items : el("li", { class: "muted" }, "No se detectó habla."));
  player.addEventListener("timeupdate", () => {
    const now = player.currentTime;
    list.querySelectorAll(".segment").forEach((item) => {
      item.classList.toggle("is-active", now >= Number(item.dataset.start) && now < Number(item.dataset.end));
    });
  });
  return el("section", { class: "card" },
    el("div", { class: "card-head" },
      el("h2", {}, "Transcripción"),
      el("span", { class: "muted small" },
        `Idioma: ${transcript.language} (${Math.round(transcript.language_probability * 100)} %) · ${transcript.engine} ${transcript.model} · ${transcript.device} ${transcript.compute_type}`)),
    el("p", { class: "notice" }, transcript.notice),
    list,
    el("p", { class: "muted small" }, "Las palabras subrayadas tienen baja confianza del modelo."));
}

function renderWord(word) {
  const uncertain = word.probability < UNCERTAIN_WORD_PROBABILITY;
  return el("span", {
    class: uncertain ? "word is-uncertain" : "word",
    title: `Confianza del modelo: ${Math.round(word.probability * 100)} %`,
  }, word.text);
}

function renderFrames(manifest, seek) {
  const summary = manifest.visual_summary || {};
  const filterKeys = Object.keys(summary).sort((a, b) => summary[b] - summary[a]);

  const figures = manifest.frames.map((frame) => {
    const detections = frame.detections || [];
    const boxes = detections.map((d) => {
      const left = d.box[0] * 100;
      const top = d.box[1] * 100;
      const width = Math.max(0, (d.box[2] - d.box[0]) * 100);
      const height = Math.max(0, (d.box[3] - d.box[1]) * 100);
      return el("div", {
        class: "detection-box",
        style: `left:${left}%; top:${top}%; width:${width}%; height:${height}%;`,
      }, el("span", { class: "detection-tag" }, `${d.label} ${Math.round(d.confidence * 100)}%`));
    });

    const pills = detections.map((d) =>
      el("span", { class: "det-pill", title: `Confianza: ${Math.round(d.confidence * 100)}%` }, `${d.label} ${Math.round(d.confidence * 100)}%`)
    );

    return el("figure", { class: "frame-card", dataset: { labels: detections.map((d) => d.label).join(",") } },
      el("button", {
        type: "button",
        class: "frame",
        onclick: () => seek(frame.timestamp_s),
        title: `Ir a ${formatClock(frame.timestamp_s)}${detections.length ? ` · ${detections.map((d) => d.label).join(", ")}` : ""}`,
      },
        el("img", { src: assetUrl(manifest.evidence_id, frame.name), alt: `Fotograma en ${formatClock(frame.timestamp_s)}`, loading: "lazy" }),
        ...boxes,
      ),
      el("figcaption", {}, formatClock(frame.timestamp_s)),
      pills.length ? el("div", { class: "frame-detections" }, ...pills) : null,
    );
  });

  const framesGrid = el("div", { class: "frames" }, figures);

  let filterBar = null;
  if (filterKeys.length > 0) {
    const chips = [
      el("button", {
        type: "button",
        class: "filter-chip is-active",
        onclick: (e) => applyFilter(null, e.currentTarget),
      }, `Todos (${manifest.frames.length})`),
      ...filterKeys.map((label) =>
        el("button", {
          type: "button",
          class: "filter-chip",
          onclick: (e) => applyFilter(label, e.currentTarget),
        }, `${label} (${summary[label]})`)
      ),
    ];
    filterBar = el("div", { class: "visual-filters", "aria-label": "Filtrar por objeto detectado" },
      el("span", { class: "muted small" }, "Filtrar por objeto detectado:"),
      ...chips,
    );

    function applyFilter(label, activeBtn) {
      filterBar.querySelectorAll(".filter-chip").forEach((chip) => chip.classList.toggle("is-active", chip === activeBtn));
      figures.forEach((fig) => {
        const labels = (fig.dataset.labels || "").split(",");
        fig.classList.toggle("is-hidden", Boolean(label && !labels.includes(label)));
      });
    }
  }

  const detectedCount = manifest.frames.reduce((acc, f) => acc + (f.detections ? f.detections.length : 0), 0);
  const headSub = detectedCount > 0
    ? `1 cada ${manifest.frame_interval_s} s · ${manifest.frames.length} fotogramas · ${detectedCount} detecciones IA (YOLOv8)`
    : `1 cada ${manifest.frame_interval_s} s · ${manifest.frames.length} fotogramas`;

  return el("section", { class: "card" },
    el("div", { class: "card-head" },
      el("h2", {}, "Fotogramas y análisis visual IA"),
      el("span", { class: "muted small" }, headSub)),
    filterBar,
    framesGrid);
}

function renderMetadata(manifest) {
  const { media, ingest } = manifest;
  const streams = media.streams.map((s) => s.codec_type === "video"
    ? `video ${s.codec_name} ${s.width}×${s.height}${s.frame_rate ? ` @ ${s.frame_rate} fps` : ""}`
    : `audio ${s.codec_name} ${s.sample_rate ?? "?"} Hz × ${s.channels ?? "?"}`).join(" · ");
  const copy = el("button", { type: "button", class: "btn-ghost small",
    onclick: (event) => navigator.clipboard.writeText(ingest.original.sha256).then(() => { event.target.textContent = "Copiado"; }) }, "Copiar");
  const rows = [
    ["Formato", media.format_long_name || media.format_name],
    ["Duración", formatClock(media.duration_s)],
    ["Tamaño", formatBytes(ingest.original.size_bytes)],
    ["Fecha de creación (metadatos)", media.creation_time ? formatDate(media.creation_time) : "No registrada en el archivo"],
    ["Pistas", streams],
    ["SHA-256 del original", el("span", { class: "hash-row" }, el("code", {}, ingest.original.sha256), copy)],
    ["Integridad", manifest.integrity_verified ? "Verificada: el original descifrado coincide con el hash de ingesta" : "No verificada"],
    ["Cifrado en reposo", "AES-256-GCM por bloques · clave derivada con Argon2id"],
  ];
  return el("section", { class: "card" },
    el("h2", {}, "Metadatos e integridad"),
    el("dl", { class: "meta" }, rows.flatMap(([term, value]) => [el("dt", {}, term), el("dd", {}, value)])));
}

function renderLineage(manifest) {
  const assets = [manifest.ingest.original, ...manifest.derivatives].map((asset) => el("li", {},
    el("strong", {}, ROLE_LABELS[asset.role]), ` · ${asset.media_type} · ${formatBytes(asset.size_bytes)}`,
    el("code", { class: "block" }, asset.sha256),
    asset.processing ? el("code", { class: "block muted" }, `${asset.processing.tool_version}\n${asset.processing.arguments.join(" ")}`) : null));
  return el("section", { class: "card" },
    el("details", {},
      el("summary", {}, "Cadena de custodia y linaje de procesamiento"),
      el("ol", { class: "lineage" }, assets),
      el("ul", { class: "notices" }, manifest.notices.map((notice) => el("li", {}, notice)))));
}

// --- Captura ---------------------------------------------------------------------------------------

function showCaptureHint(message, isError = false) {
  const hint = $("capture-hint");
  hint.textContent = message;
  hint.classList.toggle("error", isError);
}

function extensionFor(mimeType, kind) {
  if (mimeType.includes("webm")) return ".webm";
  if (mimeType.includes("ogg")) return ".ogg";
  if (mimeType.includes("mp4")) return kind === "audio" ? ".m4a" : ".mp4";
  return ".webm";
}

function setRecordingUi(active) {
  $("rec-toggle").classList.toggle("is-recording", active);
  $("rec-label").textContent = active ? "Detener y procesar" : "Iniciar grabación";
  document.querySelectorAll('input[name="rec-kind"]').forEach((input) => { input.disabled = active; });
}

async function startRecording() {
  if (!window.MediaRecorder || !navigator.mediaDevices?.getUserMedia) {
    showCaptureHint("Este navegador no permite grabar. Use Subir archivo.", true);
    return;
  }
  recorder.kind = document.querySelector('input[name="rec-kind"]:checked').value;
  try {
    recorder.stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: recorder.kind === "video" });
  } catch (error) {
    showCaptureHint(`No se pudo acceder al micrófono o la cámara: ${error.message}`, true);
    return;
  }
  const mimeType = RECORDER_TYPES[recorder.kind].find((type) => MediaRecorder.isTypeSupported(type));
  recorder.media = new MediaRecorder(recorder.stream, mimeType ? { mimeType } : undefined);
  recorder.chunks = [];
  recorder.media.ondataavailable = (event) => event.data.size && recorder.chunks.push(event.data);
  recorder.media.onstop = finishRecording;
  recorder.media.start(RECORDER_TIMESLICE_MS);
  if (recorder.kind === "video") {
    const preview = $("rec-preview");
    preview.srcObject = recorder.stream;
    preview.hidden = false;
    preview.play();
  }
  recorder.startedAt = Date.now();
  recorder.timer = setInterval(() => {
    $("rec-timer").textContent = formatClock((Date.now() - recorder.startedAt) / 1000);
  }, 250);
  setRecordingUi(true);
  showCaptureHint("Grabando… nada sale de este equipo.");
}

function stopRecording() {
  recorder.media.stop();
  recorder.stream.getTracks().forEach((track) => track.stop());
  clearInterval(recorder.timer);
}

function finishRecording() {
  const mimeType = recorder.media.mimeType || recorder.chunks[0]?.type || "";
  const blob = new Blob(recorder.chunks, { type: mimeType });
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace("T", "-").slice(0, 15);
  const preview = $("rec-preview");
  preview.srcObject = null;
  preview.hidden = true;
  $("rec-timer").textContent = "00:00";
  setRecordingUi(false);
  showCaptureHint("La grabación se envía solo a este equipo y se cifra al recibirse.");
  submitEvidence(blob, `grabacion-${stamp}${extensionFor(mimeType, recorder.kind)}`, "recording");
}

function toggleRecording() {
  if (recorder.media?.state === "recording") stopRecording();
  else startRecording();
}

function selectTab(name) {
  for (const tab of ["record", "upload"]) {
    $(`tab-${tab}`).setAttribute("aria-selected", String(tab === name));
    $(`panel-${tab}`).hidden = tab !== name;
  }
}

function wireCapture() {
  $("tab-record").addEventListener("click", () => selectTab("record"));
  $("tab-upload").addEventListener("click", () => selectTab("upload"));
  $("rec-toggle").addEventListener("click", toggleRecording);
  const input = $("file-input");
  const dropzone = $("dropzone");
  input.addEventListener("change", () => {
    if (input.files.length) submitEvidence(input.files[0], input.files[0].name, "upload");
    input.value = "";
  });
  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("is-over");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("is-over"));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-over");
    const file = event.dataTransfer.files[0];
    if (file) submitEvidence(file, file.name, "upload");
  });
  $("refresh-list").addEventListener("click", () => loadList());
}

async function init() {
  wireCapture();
  loadHealth();
  try {
    const items = await loadList();
    const latest = items.find((item) => item.status === "processed");
    if (latest) await selectEvidence(latest.evidence_id);
  } catch (error) {
    $("list-empty").hidden = false;
    $("list-empty").textContent = error.message;
  }
}

init();

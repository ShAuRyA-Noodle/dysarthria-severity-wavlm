// ATD streaming demo client (Track A scaffold).
const CHUNK_MS = 50;
const SAMPLE_RATE = 16000;

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const wordIn = document.getElementById("word");
const statusEl = document.getElementById("status");
const finalEl = document.getElementById("finalAtd");
const framesEl = document.getElementById("frames");
const latencyEl = document.getElementById("latency");

const ctx = document.getElementById("atdChart").getContext("2d");
const chart = new Chart(ctx, {
  type: "line",
  data: { labels: [], datasets: [{ label: "ATD", data: [], borderWidth: 2, pointRadius: 0 }] },
  options: {
    animation: false,
    scales: { y: { beginAtZero: true } },
  },
});

let ws = null;
let audioCtx = null;
let processor = null;
let stream = null;
let seq = 0;
const latencies = [];

function pcm16Base64(float32) {
  const buf = new ArrayBuffer(float32.length * 2);
  const view = new DataView(buf);
  for (let i = 0; i < float32.length; i++) {
    let s = Math.max(-1, Math.min(1, float32[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  let bin = "";
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bytes.byteLength; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  stopBtn.disabled = false;
  statusEl.textContent = "connecting";

  ws = new WebSocket(`ws://${location.host}/ws/atd`);
  ws.onopen = () => {
    ws.send(JSON.stringify({ type: "session_start", word: wordIn.value, sample_rate: SAMPLE_RATE, client_ts: Date.now() }));
    statusEl.textContent = "recording";
  };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "atd_update") {
      chart.data.labels.push(m.frame_idx);
      chart.data.datasets[0].data.push(m.atd);
      chart.update("none");
      framesEl.textContent = m.frame_idx;
      latencies.push(Date.now() - m.server_ts);
    } else if (m.type === "session_end") {
      finalEl.textContent = m.final_atd.toFixed(4);
      latencyEl.textContent = m.latency_p50_ms.toFixed(0);
      statusEl.textContent = "ended";
    } else if (m.type === "error") {
      statusEl.textContent = "error: " + m.msg;
    }
  };
  ws.onerror = () => { statusEl.textContent = "ws error"; };

  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
  const source = audioCtx.createMediaStreamSource(stream);
  const bufferSize = Math.round(SAMPLE_RATE * CHUNK_MS / 1000);
  processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);
  processor.onaudioprocess = (e) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const float32 = e.inputBuffer.getChannelData(0);
    ws.send(JSON.stringify({
      type: "audio_chunk",
      seq: seq++,
      pcm16_b64: pcm16Base64(float32),
      client_ts: Date.now(),
    }));
  };
  source.connect(processor);
  processor.connect(audioCtx.destination);
});

stopBtn.addEventListener("click", () => {
  startBtn.disabled = false;
  stopBtn.disabled = true;
  if (processor) processor.disconnect();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  if (audioCtx) audioCtx.close();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "session_end" }));
  }
});

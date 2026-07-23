"""Built-in conversion viewer UI.

One self-contained page (no external assets, works offline) served at /ui:
upload a document, watch SSE progress, then view the result as a rendered
document, markdown source, or response JSON. Deliberately minimal — the only
knob exposed is the pipeline mode (structured vs raw); everything else stays
at server defaults.

The HTML lives in a plain string so the module needs no package-data plumbing
and can be mirrored byte-identically between the internal and public repos.
"""

UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>shrew</title>
<style>
  :root {
    --bg: #f6f5f2; --card: #ffffff; --ink: #1c1c1a; --muted: #74716a;
    --line: #e4e2dc; --accent: #7a5c3e; --accent-ink: #fff; --err: #a33a2c;
    --ok: #3d6b46; --code-bg: #f0eeea;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #191817; --card: #211f1e; --ink: #e8e6e1; --muted: #96928a;
      --line: #35322e; --accent: #b08a5e; --accent-ink: #191817;
      --err: #d3705f; --ok: #7fae88; --code-bg: #262422;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  header {
    display: flex; align-items: baseline; gap: .75rem;
    padding: .9rem 1.4rem; border-bottom: 1px solid var(--line);
  }
  header h1 { font-size: 1.05rem; margin: 0; letter-spacing: .02em; }
  header span { color: var(--muted); font-size: .8rem; }
  main { max-width: 960px; margin: 0 auto; padding: 1.2rem 1.4rem 4rem; }

  .card {
    background: var(--card); border: 1px solid var(--line);
    border-radius: 10px; padding: 1.1rem 1.2rem; margin-bottom: 1rem;
  }
  #drop {
    border: 2px dashed var(--line); border-radius: 10px; padding: 1.6rem;
    text-align: center; color: var(--muted); cursor: pointer;
    transition: border-color .15s, color .15s;
  }
  #drop.hover, #drop:hover { border-color: var(--accent); color: var(--ink); }
  #drop strong { color: var(--ink); }
  #file { display: none; }

  .row { display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; margin-top: .9rem; }
  .modes { display: flex; gap: .25rem; background: var(--code-bg); border-radius: 8px; padding: .2rem; }
  .modes label {
    padding: .3rem .8rem; border-radius: 6px; cursor: pointer; font-size: .85rem;
    color: var(--muted);
  }
  .modes input { display: none; }
  .modes input:checked + span { color: var(--ink); font-weight: 600; }
  .modes label:has(input:checked) { background: var(--card); box-shadow: 0 1px 2px rgba(0,0,0,.12); }
  button.go {
    margin-left: auto; background: var(--accent); color: var(--accent-ink);
    border: 0; border-radius: 8px; padding: .45rem 1.3rem; font-size: .9rem;
    font-weight: 600; cursor: pointer;
  }
  button.go:disabled { opacity: .45; cursor: default; }
  #picked { font-size: .85rem; color: var(--muted); }

  #prog { display: none; }
  #prog .bar {
    height: 8px; background: var(--code-bg); border-radius: 4px; overflow: hidden;
  }
  #prog .fill {
    height: 100%; width: 0%; background: var(--accent); border-radius: 4px;
    transition: width .3s;
  }
  #prog .msg { margin-top: .5rem; font-size: .85rem; color: var(--muted); }
  #error { display: none; color: var(--err); font-size: .9rem; white-space: pre-wrap; }

  #result { display: none; }
  .tabs { display: flex; gap: .25rem; border-bottom: 1px solid var(--line); margin-bottom: 1rem; }
  .tabs button {
    background: none; border: 0; border-bottom: 2px solid transparent;
    padding: .45rem .9rem; font-size: .88rem; color: var(--muted); cursor: pointer;
  }
  .tabs button.on { color: var(--ink); border-bottom-color: var(--accent); font-weight: 600; }
  #stats { font-size: .8rem; color: var(--muted); margin-left: auto; align-self: center; }

  .pane { display: none; }
  .pane.on { display: block; }
  pre.src {
    background: var(--code-bg); border-radius: 8px; padding: 1rem;
    overflow: auto; font-size: .8rem; line-height: 1.5; white-space: pre-wrap;
    word-break: break-word;
  }

  /* rendered document */
  #doc { line-height: 1.65; }
  #doc h1 { font-size: 1.35rem; margin: 1.2rem 0 .5rem; }
  #doc h2 { font-size: 1.12rem; margin: 1.1rem 0 .4rem; }
  #doc h3, #doc h4 { font-size: 1rem; margin: 1rem 0 .35rem; }
  #doc p { margin: .55rem 0; }
  #doc img { max-width: 100%; border: 1px solid var(--line); border-radius: 6px; margin: .4rem 0; }
  #doc table { border-collapse: collapse; margin: .7rem 0; font-size: .85rem; max-width: 100%; }
  #doc th, #doc td { border: 1px solid var(--line); padding: .3rem .6rem; text-align: left; }
  #doc th { background: var(--code-bg); }
  #doc .pageno {
    margin: 1.6rem 0 .8rem; display: flex; align-items: center; gap: .8rem;
    color: var(--muted); font-size: .75rem; text-transform: uppercase; letter-spacing: .08em;
  }
  #doc .pageno::before, #doc .pageno::after { content: ""; flex: 1; border-top: 1px solid var(--line); }
  #doc .figph { color: var(--muted); font-style: italic; }
</style>
</head>
<body>
<header><h1>shrew</h1><span>document conversion viewer</span></header>
<main>
  <div class="card">
    <div id="drop"><strong>Choose a file</strong> or drop it here<div id="picked"></div></div>
    <input type="file" id="file">
    <div class="row">
      <div class="modes">
        <label><input type="radio" name="mode" value="structured" checked><span>structured</span></label>
        <label><input type="radio" name="mode" value="raw"><span>raw</span></label>
      </div>
      <button class="go" id="go" disabled>Convert</button>
    </div>
  </div>

  <div class="card" id="prog">
    <div class="bar"><div class="fill" id="fill"></div></div>
    <div class="msg" id="msg">Starting…</div>
  </div>

  <div class="card" id="error"></div>

  <div class="card" id="result">
    <div class="tabs">
      <button data-pane="doc" class="on">Document</button>
      <button data-pane="md">Markdown</button>
      <button data-pane="json">JSON</button>
      <div id="stats"></div>
    </div>
    <div class="pane on" id="pane-doc"><div id="doc"></div></div>
    <div class="pane" id="pane-md"><pre class="src" id="mdsrc"></pre></div>
    <div class="pane" id="pane-json"><pre class="src" id="jsonsrc"></pre></div>
  </div>
</main>
<script>
"use strict";
const $ = id => document.getElementById(id);
let file = null;

$("drop").onclick = () => $("file").click();
$("file").onchange = () => pick($("file").files[0]);
$("drop").ondragover = e => { e.preventDefault(); $("drop").classList.add("hover"); };
$("drop").ondragleave = () => $("drop").classList.remove("hover");
$("drop").ondrop = e => {
  e.preventDefault(); $("drop").classList.remove("hover");
  if (e.dataTransfer.files.length) pick(e.dataTransfer.files[0]);
};
function pick(f) {
  file = f || null;
  $("picked").textContent = file ? file.name + " (" + (file.size/1024).toFixed(0) + " KB)" : "";
  $("go").disabled = !file;
}

$("go").onclick = async () => {
  if (!file) return;
  $("go").disabled = true;
  $("result").style.display = "none";
  $("error").style.display = "none";
  $("prog").style.display = "block";
  setProg(0, "Uploading…");

  const fd = new FormData();
  fd.append("file", file);
  fd.append("pipeline_mode", document.querySelector('input[name=mode]:checked').value);

  try {
    const resp = await fetch("v1/convert/stream", { method: "POST", body: fd });
    if (!resp.ok) throw new Error("HTTP " + resp.status + ": " + await resp.text());
    await readSSE(resp.body);
  } catch (err) {
    fail(String(err));
  } finally {
    $("go").disabled = !file;
  }
};

function setProg(pct, msg) { $("fill").style.width = pct + "%"; $("msg").textContent = msg; }
function fail(msg) {
  $("prog").style.display = "none";
  $("error").style.display = "block";
  $("error").textContent = msg;
}

async function readSSE(body) {
  const reader = body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    // sse_starlette emits \r\n line endings (verified on the wire), so
    // normalize before splitting frames. Concatenate first: a \r\n pair can
    // straddle a chunk boundary, and the leftover \r in buf pairs up with
    // the \n that arrives in the next chunk.
    buf = (buf + dec.decode(value, { stream: true })).replace(/\r\n/g, "\n");
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      handleEvent(buf.slice(0, idx));
      buf = buf.slice(idx + 2);
    }
  }
}

function handleEvent(block) {
  let ev = "message", data = "";
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) ev = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return;
  let obj;
  try { obj = JSON.parse(data); } catch { return; }
  if (ev === "progress") setProg(obj.percent ?? 0, obj.message ?? "");
  else if (ev === "error") fail(obj.message || "conversion failed");
  else if (ev === "complete") { setProg(100, "Complete"); show(obj.result); }
}

// ── result rendering ────────────────────────────────────────────────────────

function show(res) {
  $("prog").style.display = "none";
  $("result").style.display = "block";

  const log = res.processing_log || {};
  const bits = [];
  if (log.total_pages != null) bits.push(log.total_pages + " pages");
  if (log.modality) bits.push(log.modality);
  if (log.failed_pages) bits.push(log.failed_pages + " failed");
  if (log.total_time_seconds != null) bits.push(log.total_time_seconds + "s");
  $("stats").textContent = bits.join(" · ");

  $("mdsrc").textContent = res.markdown || "";
  $("jsonsrc").textContent = JSON.stringify(res, jsonElide, 2);
  $("doc").innerHTML = renderMarkdown(res.markdown || "", res.images || []);
}

// Keep the JSON tab readable: elide base64 payloads.
function jsonElide(key, value) {
  if (key === "data" && typeof value === "string" && value.length > 64)
    return "…base64, " + value.length + " chars…";
  return value;
}

document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => {
  document.querySelectorAll(".tabs button").forEach(x => x.classList.toggle("on", x === b));
  document.querySelectorAll(".pane").forEach(p =>
    p.classList.toggle("on", p.id === "pane-" + b.dataset.pane));
});

// ── minimal markdown renderer ───────────────────────────────────────────────
// Handles exactly what the pipeline emits: headings, paragraphs, <page N>
// markers, ![alt](img:N) refs resolved against images[], and verbatim
// <table> HTML passthrough. Everything else is escaped text.

function esc(s) {
  return s.replace(/[&<>"]/g, c => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;" }[c]));
}

function inline(s, images) {
  // images first so their alt text is escaped, not re-parsed
  let out = "";
  let rest = s;
  const rx = /!\[([^\]]*)\]\(img:(\d+)\)/;
  for (;;) {
    const m = rx.exec(rest);
    if (!m) { out += esc(rest); break; }
    out += esc(rest.slice(0, m.index));
    const img = images.find(i => i.index === Number(m[2]));
    if (img && img.data)
      out += '<img src="data:image/' + (img.format || "png") + ';base64,' + img.data +
             '" alt="' + esc(m[1]) + '" title="' + esc(m[1]) + '">';
    else
      out += '<span class="figph">[Figure: ' + esc(m[1] || "untitled") + "]</span>";
    rest = rest.slice(m.index + m[0].length);
  }
  return out;
}

function renderMarkdown(md, images) {
  const out = [];
  const lines = md.split("\n");
  let para = [];
  let tableBuf = null;

  const flush = () => {
    if (para.length) { out.push("<p>" + inline(para.join(" "), images) + "</p>"); para = []; }
  };

  for (const line of lines) {
    if (tableBuf !== null) {
      tableBuf.push(line);
      if (/<\/table>/i.test(line)) { out.push(tableBuf.join("\n")); tableBuf = null; }
      continue;
    }
    const t = line.trim();
    let m;
    if ((m = t.match(/^<page (\d+)>$/)))       { flush(); out.push('<div class="pageno">page ' + m[1] + "</div>"); }
    else if (/^<\/page \d+>$/.test(t))          { flush(); }
    else if (/^<table/i.test(t)) {
      flush();
      if (/<\/table>/i.test(t)) out.push(t);
      else tableBuf = [line];
    }
    else if ((m = t.match(/^(#{1,6})\s+(.*)$/))) {
      flush();
      const lvl = Math.min(m[1].length, 6);
      out.push("<h" + lvl + ">" + inline(m[2], images) + "</h" + lvl + ">");
    }
    else if (t === "") flush();
    else para.push(t);
  }
  flush();
  if (tableBuf) out.push(esc(tableBuf.join("\n")));  // unterminated table: show as text
  return out.join("\n");
}
</script>
</body>
</html>
"""

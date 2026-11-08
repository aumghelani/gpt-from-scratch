"""Single-file web UI. Model output is only ever inserted with textContent (no markup injection)."""

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GPT From Scratch</title>
<style>
  :root { --bg:#fafaf7; --fg:#1d1d1b; --muted:#6b6b66; --card:#fff; --line:#e4e3dd; --accent:#2f5d50; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#161715; --fg:#ecebe6; --muted:#9a9a93; --card:#1f201e; --line:#33342f; --accent:#7fb8a4; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 ui-sans-serif,system-ui,sans-serif; }
  main { max-width:760px; margin:0 auto; padding:24px 16px 120px; }
  h1 { font-size:20px; margin:0 0 4px; } .sub { color:var(--muted); font-size:14px; margin-bottom:20px; }
  .msg { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px;
         margin:10px 0; white-space:pre-wrap; word-wrap:break-word; }
  .msg.user { border-color:var(--accent); } .role { font-size:12px; color:var(--muted); margin-bottom:4px; }
  form { position:fixed; left:0; right:0; bottom:0; background:var(--bg); border-top:1px solid var(--line);
         padding:12px 16px; } .row { max-width:760px; margin:0 auto; display:flex; gap:8px; }
  textarea { flex:1; resize:none; padding:10px; border-radius:8px; border:1px solid var(--line);
             background:var(--card); color:var(--fg); font:inherit; }
  button { padding:0 14px; border-radius:8px; border:1px solid var(--line); background:var(--card);
           color:var(--fg); font:inherit; cursor:pointer; min-height:44px; }
  button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
  button:disabled { opacity:.5; cursor:default; }
  .ctl { max-width:760px; margin:8px auto 0; display:flex; gap:16px; flex-wrap:wrap; font-size:13px; color:var(--muted); }
  input[type=range] { vertical-align:middle; }
</style>
</head>
<body>
<main>
  <h1>GPT From Scratch</h1>
  <div class="sub" id="status">connecting…</div>
  <div id="log" aria-live="polite"></div>
</main>
<form id="form">
  <div class="row">
    <textarea id="input" rows="2" placeholder="Type a prompt…" maxlength="8000" required></textarea>
    <button class="primary" id="send" type="submit">Send</button>
    <button id="stop" type="button" disabled>Stop</button>
    <button id="regen" type="button" disabled>Redo</button>
  </div>
  <div class="ctl">
    <label id="modelbox" hidden>model <select id="model"></select></label>
    <label>temperature <input id="temp" type="range" min="0" max="2" step="0.1" value="0.8">
      <span id="tempv">0.8</span></label>
    <label>max tokens <input id="maxnew" type="number" min="1" max="1024" value="200" style="width:70px"></label>
  </div>
</form>
<script>
const $ = (id) => document.getElementById(id);
let chatMode = false, history = [], controller = null, lastPrompt = "", models = [];

function bubble(role, text) {
  const box = document.createElement("div");
  box.className = "msg " + role;
  const label = document.createElement("div");
  label.className = "role";
  label.textContent = role;
  const body = document.createElement("div");
  body.textContent = text;
  box.append(label, body);
  $("log").append(box);
  box.scrollIntoView({block: "end"});
  return body;
}

function applyModel() {
  const m = models.find((x) => x.id === $("model").value);
  if (!m) return;
  chatMode = m.chat;
  $("status").textContent = (chatMode ? "chat mode" : "text-completion mode") + " · " +
    m.params.toLocaleString() + " parameters · context " + m.block_size + " · " + m.checkpoint;
}

async function init() {
  try {
    models = await (await fetch("/v1/models")).json();
    for (const m of models) {
      const option = document.createElement("option");
      option.value = m.id;
      option.textContent = m.id + (m.chat ? " (chat)" : "");
      $("model").append(option);
    }
    $("modelbox").hidden = models.length < 2;
    applyModel();
  } catch (e) { $("status").textContent = "server unreachable"; }
}

async function run(prompt) {
  lastPrompt = prompt;
  const out = bubble(chatMode ? "assistant" : "model", "");
  controller = new AbortController();
  $("send").disabled = true; $("stop").disabled = false; $("regen").disabled = true;
  const common = {max_new_tokens: Number($("maxnew").value), temperature: Number($("temp").value), stream: true,
                  model: $("model").value || null};
  const url = chatMode ? "/v1/chat" : "/v1/generate";
  const body = chatMode ? {...common, messages: [...history, {role: "user", content: prompt}]}
                        : {...common, prompt};
  let reply = "";
  try {
    const res = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"},
                                  body: JSON.stringify(body), signal: controller.signal});
    if (!res.ok) { out.textContent = "error " + res.status; return; }
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      const events = buf.split("\\n\\n"); buf = events.pop();
      for (const ev of events) {
        const data = ev.replace(/^data: /, "");
        if (data === "[DONE]") continue;
        reply += JSON.parse(data).token;
        out.textContent = reply;
      }
    }
    if (chatMode) history = [...history, {role: "user", content: prompt}, {role: "assistant", content: reply}];
  } catch (e) {
    if (e.name !== "AbortError") out.textContent = reply + "\\n[connection error]";
  } finally {
    $("send").disabled = false; $("stop").disabled = true; $("regen").disabled = false; controller = null;
  }
}

$("form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("input").value.trim();
  if (!text) return;
  bubble("user", text); $("input").value = "";
  run(text);
});
$("input").addEventListener("keydown", (e) => {   // Enter sends; Shift+Enter inserts a newline
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("form").requestSubmit(); }
});
$("stop").addEventListener("click", () => controller && controller.abort());
$("regen").addEventListener("click", () => {
  if (!lastPrompt) return;
  const log = $("log");
  if (log.lastElementChild) log.lastElementChild.remove();           // drop the old reply bubble
  if (chatMode && history.length >= 2) history = history.slice(0, -2); // and its history entries
  run(lastPrompt);
});
$("model").addEventListener("change", () => {      // a different model starts a fresh conversation
  history = []; lastPrompt = ""; $("log").replaceChildren(); applyModel();
});
$("temp").addEventListener("input", () => { $("tempv").textContent = $("temp").value; });
init();
</script>
</body>
</html>
"""

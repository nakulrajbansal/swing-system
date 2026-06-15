"use strict";
const $ = (id) => document.getElementById(id);
const val = (id) => $(id).value;
let running = false, es = null;

// ---- navigation ----
document.querySelectorAll(".nav a").forEach(a => a.onclick = () => show(a.dataset.page));
function show(page) {
  document.querySelectorAll(".nav a").forEach(a => a.classList.toggle("active", a.dataset.page === page));
  document.querySelectorAll(".page").forEach(p => p.classList.toggle("active", p.id === "page-" + page));
  if (page === "performance") loadPerformance();
  if (page === "learning") loadLearning();
  if (page === "settings") loadSettings();
}

// ---- console ----
function logLine(s) {
  const c = $("console"), div = document.createElement("div");
  const t = s.trim();
  if (t && /^[#=\-]{8,}$/.test(t)) { div.textContent = "─".repeat(60); div.className = "rule"; }
  else { div.textContent = s; div.className = tagFor(s); }
  c.appendChild(div); c.scrollTop = c.scrollHeight;
}
function tagFor(s) {
  const l = s.toLowerCase();
  if (l.includes("[error]") || l.includes("[blocked]") || l.includes("[risk]")) return "err";
  if (l.includes("[warn") || l.includes("warning") || l.includes("[note]") || l.includes("[hint]") || l.includes("[stop")) return "warn";
  if (l.includes("hidden gem") || l.includes("hidden-gem")) return "gem";
  if (l.includes("recommend buy") || l.startsWith("[done]") || l.includes("verdict: buy")) return "ok";
  if (s.trimStart().startsWith("▸")) return "agent";
  if (s.startsWith("===") || s.startsWith("####") || s.toUpperCase().slice(0, 14).includes("ANALYST")) return "hl";
  if (s.startsWith("[") || s.startsWith("  -") || s.startsWith("  +")) return "dim";
  return "";
}
function copyConsole() { navigator.clipboard.writeText($("console").innerText); toast("Copied"); }
function toast(msg) { const t = $("toast"); t.textContent = msg; t.classList.add("show"); setTimeout(() => t.classList.remove("show"), 1800); }

// ---- running flows ----
async function run(action, opts = {}) {
  if (running) { toast("a run is already in progress"); return; }
  setRunning(true, action);
  logLine("\n=== starting: " + action + " ===");
  let r;
  try {
    r = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, universe: opts.universe || null, ticker: opts.ticker || null }) });
  } catch (e) { logLine("[error] " + e); setRunning(false); return; }
  if (!r.ok) { logLine("[error] " + (await r.text())); setRunning(false); return; }
  const { run_id } = await r.json();
  es = new EventSource("/api/stream/" + run_id);
  es.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.kind === "log") logLine(m.payload);
    else if (m.kind === "result") { if (m.payload && m.payload.recommendations) loadOrders(); }
    else if (m.kind === "done") { es.close(); es = null; setRunning(false); loadOrders(); loadWatchlist(); }
  };
  es.onerror = () => { if (es) { es.close(); es = null; } setRunning(false); };
}
function setRunning(on, action) {
  running = on;
  $("spin").classList.toggle("on", on);
  $("stopBtn").disabled = !on;
  $("statusText").textContent = on ? (action || "running") + "…" : "ready";
}
async function stopRun() { await fetch("/api/stop", { method: "POST" }); logLine("[stop] stopping at the next step…"); }

// ---- chips / env ----
async function loadConfig() {
  const c = await (await fetch("/api/config")).json();
  const chip = (txt, on) => `<span class="chip ${on ? "on" : ""}">${txt}</span>`;
  $("chips").innerHTML =
    chip("DATA · " + (c.data_source === "live" ? "LIVE" : "SYNTHETIC"), c.data_source === "live") +
    chip("AGENTS · " + (c.use_llm_agents ? "LLM" : "DETERMINISTIC"), c.use_llm_agents) +
    chip("SIZING · " + (c._has_alpaca ? "REAL ACCOUNT" : "CONFIGURED"), c._has_alpaca);
  const live = c.alpaca_env === "live";
  const b = $("envBadge");
  b.className = "badge " + (live ? "live" : "paper");
  b.textContent = live ? "⚠ LIVE" : "PAPER";
  return c;
}

// ---- orders ----
async function loadOrders() {
  const { recommendations } = await (await fetch("/api/orders")).json();
  const el = $("ordersCard");
  if (!recommendations || !recommendations.length) {
    el.innerHTML = '<span class="faint small">No BUY recommendations from the last run.</span>'; return;
  }
  let h = "<table><tr><th>symbol</th><th>conv</th><th>P(win)</th><th>ref</th><th>qty</th><th>limit</th><th>bracket</th><th></th></tr>";
  recommendations.forEach((r, i) => {
    const pw = r.p_win ? Math.round(r.p_win * 100) + "%" : "-";
    const qty = r.shares_at_ref_equity || r.affordable_qty || 0;
    const limit = r.suggested_entry || r.entry || "";
    const gem = r.hidden_gem ? ' <span class="tag">◆</span>' : "";
    const held = r.already_held ? ' <span class="tag" style="color:var(--warn)">●</span>' : "";
    h += `<tr><td>${r.symbol}${gem}${held}</td><td>${(r.conviction ?? 0).toFixed?.(2) ?? r.conviction}</td>`
      + `<td>${pw}</td><td class="muted">~$${r.entry}</td>`
      + `<td><input id="q${i}" type="number" value="${qty}"></td>`
      + `<td><input id="p${i}" type="number" value="${limit}" style="width:90px"></td>`
      + `<td><input id="b${i}" type="checkbox" ${(r.stop && r.target) ? "checked" : ""}></td>`
      + `<td><button class="accent" onclick='placeOrder(${i})'>Place</button></td></tr>`;
  });
  el.innerHTML = h + "</table>";
  el._recs = recommendations;
}
async function placeOrder(i) {
  const r = $("ordersCard")._recs[i];
  const qty = parseInt($("q" + i).value), price = parseFloat($("p" + i).value);
  if (!(qty > 0)) { toast("quantity must be > 0"); return; }
  if (!confirm(`Submit BUY ${qty} ${r.symbol} (limit $${price})?`)) return;
  const body = { symbol: r.symbol, qty, order_type: "limit", limit_price: price,
    stop: r.stop, target: r.target, attach_bracket: $("b" + i).checked, ref_price: r.entry };
  const res = await (await fetch("/api/order", { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
  (res.log || []).forEach(logLine);
  toast(res.ok ? "✓ order sent" : "✗ order failed");
}

// ---- watchlist ----
async function loadWatchlist() {
  const { items } = await (await fetch("/api/watchlist")).json();
  $("watchCard").innerHTML = (items && items.length)
    ? items.map(it => `<span class="small">${it.symbol} (${it.pullback_target ? "dip to " + it.pullback_target : "break " + it.breakout_level})</span>`).join("&nbsp;&nbsp;&nbsp;")
    : '<span class="faint small">empty — screens add WATCH-tier ideas and pullback calls automatically.</span>';
}

// ---- performance ----
async function loadPerformance() {
  const p = await (await fetch("/api/performance")).json();
  drawEquity(p.equity_curve || [1]);
  let t = `SCORED CALLS  ${p.n}   |   hit ${p.hit_rate}%   |   avg ${p.avg_return >= 0 ? "+" : ""}${p.avg_return}%\n\nCALIBRATION (conviction band → realized win rate)\n`;
  (p.calibration || []).forEach(b => {
    t += `  ${b.lo.toFixed(2)}-${Math.min(b.hi, 1).toFixed(2)}: n=${b.n}  ${b.win_rate_pct ?? "-"}${b.win_rate_pct != null ? "%" : ""}\n`;
  });
  t += "\nLENS COHORTS\n";
  for (const [k, label] of [["hidden_gem", "hidden-gem"], ["core", "core"], ["moat_bullish", "moat-bullish"]]) {
    const co = (p.cohorts || {})[k] || {};
    if (co.n) t += `  ${label.padEnd(12)} n=${co.n}  hit ${co.win_rate_pct}%  avg ${co.avg_return_pct >= 0 ? "+" : ""}${co.avg_return_pct}%\n`;
  }
  t += "\nRECENT CLOSED CALLS\n";
  (p.recent || []).forEach(r => {
    t += `  ${r.date}  ${(r.symbol + (r.hidden_gem ? " ◆" : "")).padEnd(9)} ${r.return_pct >= 0 ? "+" : ""}${r.return_pct}%  (${r.outcome}, conv ${r.conviction})\n`;
  });
  if (!p.n) t = `No scored calls yet — ${p.open} open recommendation(s) awaiting their exit windows.`;
  $("perfText").textContent = t;
}
function drawEquity(eq) {
  const cv = $("equity"); cv.width = cv.clientWidth || 1000;
  const ctx = cv.getContext("2d"), W = cv.width, H = cv.height, pad = 16;
  ctx.clearRect(0, 0, W, H);
  if (eq.length < 3) { ctx.fillStyle = "#8893a4"; ctx.font = "13px sans-serif";
    ctx.fillText("The curve starts once recommendations mature and are scored.", pad, H / 2); return; }
  const lo = Math.min(...eq, 1), hi = Math.max(...eq, 1), span = (hi - lo) * 1.12 || 0.05;
  const x = i => pad + i * (W - 2 * pad) / (eq.length - 1);
  const y = v => H - pad - (v - (lo - span * .06)) * (H - 2 * pad) / (span);
  ctx.strokeStyle = "#39414f"; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(pad, y(1)); ctx.lineTo(W - pad, y(1)); ctx.stroke();
  ctx.setLineDash([]); ctx.strokeStyle = "#4ccb96"; ctx.lineWidth = 2; ctx.beginPath();
  eq.forEach((v, i) => i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))); ctx.stroke();
}

// ---- learning ----
async function loadLearning() {
  const l = await (await fetch("/api/learning")).json();
  $("learnText").textContent = (l.lessons || "") + "\n\n" + "─".repeat(50) + "\n" + (l.ledger || "");
}

// ---- settings ----
const SETTING_FIELDS = [
  ["anthropic_api_key", "Anthropic API key", "password"],
  ["alpaca_key_id", "Alpaca key id", "password"],
  ["alpaca_secret", "Alpaca secret", "password"],
  ["alpaca_env", "Alpaca environment", "select:paper,live"],
  ["edgar_user_agent", "EDGAR User-Agent (email)", "text"],
  ["data_source", "Data source", "select:synthetic,live"],
  ["screen_index", "Default screen universe", "text"],
  ["use_llm_agents", "Use real AI agents", "bool"],
  ["place_orders", "Auto-place approved orders", "bool"],
  ["enable_live_trading", "Allow LIVE (real-money) trading", "bool"],
  ["learn_from_runs", "Learn from closed trades", "bool"],
  ["self_tune_weights", "Self-tune the screen nightly", "bool"],
];
async function loadSettings() {
  const c = await loadConfig();
  const f = $("settingsForm"); f.innerHTML = "";
  SETTING_FIELDS.forEach(([k, label, type]) => {
    const lab = document.createElement("label"); lab.textContent = label; f.appendChild(lab);
    let inp;
    if (type === "bool") { inp = document.createElement("input"); inp.type = "checkbox"; inp.checked = !!c[k]; }
    else if (type.startsWith("select")) {
      inp = document.createElement("select");
      type.split(":")[1].split(",").forEach(o => { const op = document.createElement("option"); op.value = op.textContent = o; inp.appendChild(op); });
      inp.value = c[k];
    } else { inp = document.createElement("input"); inp.type = (type === "password") ? "password" : "text"; inp.value = c[k] || ""; }
    inp.id = "set_" + k; f.appendChild(inp);
  });
}
async function saveSettings() {
  const patch = {};
  SETTING_FIELDS.forEach(([k, , type]) => {
    const el = $("set_" + k);
    patch[k] = (type === "bool") ? el.checked : el.value;
  });
  await fetch("/api/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(patch) });
  $("saveMsg").textContent = "✓ saved"; setTimeout(() => $("saveMsg").textContent = "", 3000);
  loadConfig();
}

// ---- init ----
loadConfig(); loadOrders(); loadWatchlist();

// Research Desk front end. All text from the API, which includes untrusted web
// content, is inserted with textContent, never innerHTML.

const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c == null) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

function safeUrl(url) {
  try {
    const u = new URL(url);
    return u.protocol === "https:" || u.protocol === "http:" ? u.href : null;
  } catch { return null; }
}

const STAGES = ["Planning", "Searching", "Reading papers", "Sanitizing", "Writing claims", "Checking quotes"];
const STATUS_LABEL = { completed: "Answered", partial: "Partial", abstained: "Abstained", error: "Error" };
let lastAnswer = null;
let timer = null;

// ---------- config ----------
fetch("/api/config").then((r) => r.json()).then((cfg) => {
  const llm = cfg.llm_chain.length ? cfg.llm_chain.join("  →  ") : `not configured (${cfg.llm_error})`;
  $("chain").textContent = `Models tried in order: ${llm}.  Search: ${cfg.search_chain.join(" → ")}` +
    `${cfg.paper_search ? " + arXiv/Semantic Scholar" : ""}.  Budget per run: ${cfg.budgets.tool_calls} tool calls, ` +
    `${cfg.budgets.llm_calls} model calls, ${cfg.budgets.wall_clock_s}s.`;
}).catch(() => {});
$("issue").textContent = new Date().toISOString().slice(0, 10);

// ---------- form ----------
document.querySelectorAll(".chip").forEach((b) =>
  b.addEventListener("click", () => { $("q").value = b.dataset.q; $("q").focus(); }));

$("opt-offline").addEventListener("change", (e) => {
  if (e.target.checked) $("q").value = "What is Python?";
});

$("q").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) $("ask").requestSubmit();
});

$("ask").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("q").value.trim();
  if (question.length < 3) return;
  setBusy(true);
  try {
    const res = await fetch("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        offline: $("opt-offline").checked,
        defenses: $("opt-defenses").checked,
        baseline: $("opt-baseline").checked,
      }),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `Request failed (${res.status})`);
    const data = await res.json();
    render(data);
    $("status").hidden = true;
  } catch (err) {
    showStatus(`Couldn't finish the research: ${err.message}`, true);
  } finally {
    setBusy(false);
  }
});

function setBusy(busy) {
  $("submit").disabled = busy;
  $("submit").textContent = busy ? "Researching…" : "Research it";
  clearInterval(timer);
  if (!busy) return;
  const started = Date.now();
  const steps = el("div", { class: "status__steps" }, STAGES.map((s) => el("span", {}, s)));
  const line = el("div", {}, "Working. Live runs usually take 15–30 seconds.");
  $("status").className = "status";
  $("status").replaceChildren(line, steps);
  $("status").hidden = false;
  // Elapsed-time pacing only: the server returns once, at the end.
  timer = setInterval(() => {
    const s = Math.floor((Date.now() - started) / 1000);
    line.textContent = `Working · ${s}s. Live runs usually take 15–30 seconds.`;
    const lit = Math.min(STAGES.length, Math.floor(s / 3) + 1);
    [...steps.children].forEach((c, i) => c.classList.toggle("on", i < lit));
  }, 500);
}

function showStatus(msg, isError) {
  $("status").className = isError ? "status status--error" : "status";
  $("status").textContent = msg;
  $("status").hidden = false;
}

// ---------- render ----------
function render({ answer, baseline, trace }) {
  lastAnswer = answer;
  const status = answer.execution_status;

  const stamp = $("stamp");
  stamp.className = `stamp stamp--${status}`;
  stamp.textContent = STATUS_LABEL[status] || status;
  $("question-echo").textContent = answer.question;
  $("short-answer").textContent = answer.short_answer;
  $("abstain-reason").hidden = !answer.abstention_reason;
  $("abstain-reason").textContent = answer.abstention_reason
    ? `The agent chose not to answer: ${answer.abstention_reason}` : "";

  renderScale(answer);

  // Sources become numbered footnotes; claims link to them.
  const urls = [];
  const fnFor = (url) => { let i = urls.indexOf(url); if (i < 0) { urls.push(url); i = urls.length - 1; } return i + 1; };
  const srcMeta = {};

  const findings = answer.verified_findings || [];
  $("n-verified").textContent = String(findings.length);
  $("findings").replaceChildren(...(findings.length ? findings.map((f) => {
    const sups = f.supporting_evidence.map((ev) => {
      srcMeta[ev.url] = ev;
      const n = fnFor(ev.url);
      return el("sup", {}, el("a", { href: `#fn-${n}`, "aria-label": `Source ${n}` }, `[${n}]`));
    });
    return el("li", {},
      el("div", {},
        el("p", { class: "claim__text" }, f.claim.claim_text, ...sups),
        ...f.claim.verbatim_quotes.map((q) => el("blockquote", { class: "claim__quote" }, `“${q}”`)),
        el("div", { class: "claim__meta" },
          el("span", { class: "stamp stamp--verified" }, "Verified"),
          el("span", {}, `${f.verification_checks_passed.length} checks passed`))));
  }) : [el("li", { class: "empty" }, "No claim survived verification.")]));

  const rejected = answer.rejected_claims || [];
  $("rejected-wrap").hidden = rejected.length === 0;
  $("n-rejected").textContent = String(rejected.length);
  $("rejected").replaceChildren(...rejected.map((r) =>
    el("li", {},
      el("div", {},
        el("p", { class: "claim__text" }, r.claim.claim_text),
        el("ul", { class: "reasons" }, r.rejection_reasons.map((x) => el("li", {}, x))),
        el("div", { class: "claim__meta" }, el("span", { class: "stamp stamp--struck" }, "Struck"))))));

  $("sources").replaceChildren(...urls.map((u, i) => {
    const ev = srcMeta[u] || {};
    const href = safeUrl(u);
    const title = ev.title || ev.domain || u;
    return el("li", { id: `fn-${i + 1}` },
      href ? el("a", { href, target: "_blank", rel: "noopener noreferrer" }, title) : title,
      el("span", { class: "src-domain" }, `${ev.domain || ""} · ${ev.provider || ""}`));
  }));
  $("no-sources").hidden = urls.length > 0;

  $("baseline-wrap").hidden = !baseline;
  if (baseline) $("baseline-text").textContent = baseline.short_answer;

  fillList("limitations", [...(answer.limitations || []), ...(answer.assumptions || [])]);
  fillList("next-steps", answer.suggested_next_steps || []);
  renderProcess(answer, trace || []);

  $("report").hidden = false;
  $("report").scrollIntoView({ block: "start" });
}

function renderScale(answer) {
  const c = Math.max(0, Math.min(1, answer.confidence_score || 0));
  $("conf-value").textContent = c.toFixed(2);
  $("conf-marker").style.left = `${c * 100}%`;
  $("threshold").style.left = "30%";
  $("scale-track").setAttribute("aria-label", `Confidence ${c.toFixed(2)} of 1. Abstains below 0.30.`);

  // "Confidence: 0.84 (...). Verification rate: 70% (7/10 claims); Coverage ratio: 88%; ..."
  const parts = (answer.confidence_explanation || "").split(/[.;]\s+/).slice(1);
  const rows = parts.map((p) => p.split(/:\s*/)).filter((kv) => kv.length === 2 && kv[1].trim());
  $("signals").replaceChildren(...rows.slice(0, 6).map(([k, v]) =>
    el("div", {}, el("dt", {}, k.trim()), el("dd", {}, v.trim().replace(/\.$/, "")))));
}

function fillList(id, items) {
  $(id).replaceChildren(...(items.length ? items : ["None noted."]).map((t) => el("li", {}, t)));
}

function renderProcess(answer, trace) {
  const m = answer.run_metadata || {};
  $("run-meta").textContent = m.run_id
    ? `${(m.wall_clock_seconds || 0).toFixed(1)}s · ${m.llm_calls} model calls · ${m.tool_calls} tool calls · ` +
      `${m.llm_provider || "?"} / ${m.llm_model || "?"}`
    : "";

  const planEvent = trace.find((e) => e.event_type === "plan");
  const plan = planEvent && planEvent.data && planEvent.data.plan;
  $("plan").replaceChildren(...(plan ? [
    el("p", {}, el("strong", {}, "Plan "), `(${plan.question_type})`),
    el("ol", {}, (plan.steps || []).map((s) =>
      el("li", {}, s.description, " ", ...(s.search_queries || []).map((q) => [el("code", {}, q), " "])))),
  ] : []));

  $("trace").replaceChildren(...trace.map((e) =>
    el("li", { class: e.event_type === "error" ? "err" : "" },
      el("span", { class: "t" }, (e.timestamp || "").slice(11, 22)),
      el("span", {}, e.stage),
      el("span", { class: "e" }, `${e.event_type}  ${JSON.stringify(e.data).slice(0, 180)}`))));
}

$("download").addEventListener("click", () => {
  if (!lastAnswer) return;
  const blob = new Blob([JSON.stringify(lastAnswer, null, 2)], { type: "application/json" });
  const a = el("a", { href: URL.createObjectURL(blob), download: `answer_${lastAnswer.run_metadata?.run_id || "run"}.json` });
  document.body.append(a); a.click(); a.remove();
});

// ---------- injection bench ----------
$("bench-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const res = await fetch("/api/security", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: $("bench-text").value, is_html: $("bench-html").checked }),
  });
  const r = await res.json();
  const verdict = r.quarantined ? "quarantined" : (r.flags.length ? "flagged" : "clean");
  $("bench-stamp").className = `stamp stamp--${verdict}`;
  $("bench-stamp").textContent = verdict;
  // Flags arrive as "family: matched context"; the family name is the readable part.
  const families = [...new Set(r.flags.map((f) => String(Array.isArray(f) ? f[0] : f).split(":")[0].replace(/_/g, " ")))];
  const flags = families.join(", ");
  $("bench-summary").textContent = verdict === "clean"
    ? "Nothing suspicious found."
    : `${r.flags.length} pattern${r.flags.length === 1 ? "" : "s"} matched (${flags}). Trust reduced by ${r.trust_reduction.toFixed(2)}.`;
  $("bench-clean").textContent = r.cleaned_text || "(nothing: the whole text was quarantined)";
  $("bench-out").hidden = false;
});

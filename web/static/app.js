// The professor's page: choose a brief, watch the run, read the report.
(function () {
  "use strict";

  const STAGES = [
    ["reading", "Reading the brief"],
    ["modeling", "Building the model"],
    ["solving", "Solving"],
    ["checking", "Checking the numbers"],
    ["independent", "Independent check"],
    ["report", "Writing the report"],
  ];
  const LABEL = Object.fromEntries(STAGES);
  const MIN_ACTIVE_MS = 1800; // every stage stays on screen long enough to be seen
  const MAX_BYTES = 200000;

  const STATUS = {
    passed: { badge: "Verified", tone: "ok", title: "Verified" },
    failed: { badge: "Not verified", tone: "bad", title: "Not verified" },
    refused: { badge: "Couldn't analyze", tone: "warn", title: "Couldn't analyze this brief" },
    busy: { badge: "Service busy", tone: "warn", title: "The AI service is busy" },
    error: { badge: "Error", tone: "bad", title: "Something went wrong" },
    interrupted: { badge: "Interrupted", tone: "warn", title: "This analysis was interrupted" },
    running: { badge: "Running", tone: "live", title: "Still running" },
  };

  const $ = (id) => document.getElementById(id);
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const BADGE_CLASS = { ok: "ok", bad: "bad", warn: "warn", live: "live" };

  let chosen = null; // { file, text }
  let current = null; // the run this page started, while it is going or just finished

  // ---------------------------------------------------------------- views

  function show(view) {
    ["upload", "run", "report"].forEach((name) => {
      $("view-" + name).hidden = name !== view;
    });
    window.scrollTo(0, 0);
  }

  function route() {
    const hash = location.hash || "#/";
    const match = hash.match(/^#\/run\/([0-9a-f-]+)$/);
    markCurrent(match ? match[1] : null);
    if (match) {
      if (current && current.id === match[1] && !current.finished) {
        location.replace("#/running");
        return;
      }
      openRun(match[1]);
    } else if (hash === "#/running" && current) {
      show("run");
    } else {
      if (current && current.finished) {
        current = null;
        resetUpload();
      }
      show("upload");
    }
  }

  function resetUpload() {
    chosen = null;
    $("preview").innerHTML = "";
    $("preview-wrap").hidden = true;
    $("drop").hidden = false;
    uploadError("");
  }

  // ------------------------------------------------------------ past runs

  function when(iso) {
    const date = new Date(iso);
    if (isNaN(date)) return "";
    return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(date);
  }

  async function loadRuns() {
    let runs = [];
    try {
      const response = await fetch("/api/runs");
      if (response.ok) runs = await response.json();
    } catch (e) {
      return;
    }
    const list = $("runs");
    list.textContent = "";
    $("runs-empty").hidden = runs.length > 0;
    runs.forEach((run) => {
      const status = STATUS[run.status] || STATUS.error;
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = current && current.id === run.id && !current.finished ? "#/running" : "#/run/" + run.id;
      a.dataset.run = run.id;
      const title = document.createElement("span");
      title.textContent = run.title || "Untitled brief";
      const meta = document.createElement("span");
      meta.className = "run-when";
      const time = document.createElement("span");
      time.textContent = when(run.created);
      const badge = document.createElement("span");
      badge.className = "badge " + BADGE_CLASS[status.tone];
      badge.textContent = status.badge;
      meta.append(time, badge);
      a.append(title, meta);
      li.append(a);
      list.append(li);
    });
    const match = (location.hash || "").match(/^#\/run\/([0-9a-f-]+)$/);
    markCurrent(match ? match[1] : null);
  }

  function markCurrent(runId) {
    document.querySelectorAll("#runs a").forEach((a) => {
      if (runId && a.dataset.run === runId) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
  }

  // --------------------------------------------------------------- upload

  function uploadError(message) {
    $("upload-error").textContent = message;
    $("upload-error").hidden = !message;
  }

  async function choose(file) {
    uploadError("");
    if (!file) return;
    if (!/\.md$/i.test(file.name)) {
      uploadError("That isn't a brief file. Choose a .md file.");
      return;
    }
    if (file.size > MAX_BYTES) {
      uploadError("That file is too large for a brief (over 200 KB).");
      return;
    }
    const text = await file.text();
    if (!text.trim()) {
      uploadError("That brief is empty.");
      return;
    }
    chosen = { file, text };
    $("preview-name").textContent = file.name;
    window.renderMarkdown(text, $("preview"), { foldCode: true });
    $("preview-wrap").hidden = false;
    $("drop").hidden = true;
    $("run").focus();
  }

  function wireUpload() {
    const drop = $("drop");
    const input = $("file");
    drop.addEventListener("click", () => input.click());
    $("choose-again").addEventListener("click", () => input.click());
    input.addEventListener("change", () => {
      choose(input.files[0]);
      input.value = "";
    });
    ["dragenter", "dragover"].forEach((type) =>
      drop.addEventListener(type, (e) => {
        e.preventDefault();
        drop.classList.add("over");
      })
    );
    ["dragleave", "drop"].forEach((type) => drop.addEventListener(type, () => drop.classList.remove("over")));
    drop.addEventListener("drop", (e) => {
      e.preventDefault();
      choose(e.dataTransfer.files[0]);
    });
    $("run").addEventListener("click", () => chosen && startRun(chosen));
  }

  // ------------------------------------------------------------- the run

  function stageIcon(status) {
    if (status === "done") return '<svg class="mark" viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8.5l3 3 7-7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    if (status === "failed") return '<svg class="mark" viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
    if (status === "active") return '<svg class="mark" viewBox="0 0 16 16" aria-hidden="true"><g class="spin"><path d="M8 2a6 6 0 1 1-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></g></svg>';
    return '<svg class="mark" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>';
  }

  function resetRunView(title) {
    $("run-title").textContent = title;
    $("stages").innerHTML = STAGES.map(
      ([key, label]) =>
        '<li class="stage waiting" id="stage-' + key + '">' + stageIcon("waiting") +
        '<span class="what">' + label + '</span><span class="head"></span></li>'
    ).join("");
    $("stage-panel").innerHTML = window.stageScene("reading", {}, []);
    $("explain-title").textContent = "Starting";
    $("explain-body").textContent = "Sending your brief to the agent.";
    $("explain-note").hidden = true;
    $("run-outcome").hidden = true;
  }

  function startClock() {
    const started = Date.now();
    const tick = () => {
      const s = Math.round((Date.now() - started) / 1000);
      $("run-clock").textContent = "Elapsed " + Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0") + ". Keep this page open until it finishes.";
    };
    tick();
    return setInterval(tick, 500);
  }

  async function startRun(brief) {
    uploadError("");
    $("run").disabled = true;
    const form = new FormData();
    form.append("brief", brief.file, brief.file.name);
    let response;
    try {
      response = await fetch("/api/runs", { method: "POST", body: form });
    } catch (e) {
      $("run").disabled = false;
      uploadError("Couldn't reach the server. Check your connection and try again.");
      return;
    }
    $("run").disabled = false;
    if (!response.ok) {
      let detail = "";
      try {
        detail = (await response.json()).detail;
      } catch (e) {}
      uploadError(detail || "The server couldn't start the analysis.");
      return;
    }
    current = { id: null, finished: false, stages: {}, active: null, lastChange: 0, queue: [], pumping: false, clock: null };
    resetRunView(brief.file.name);
    location.hash = "#/running";
    show("run");
    readStream(response.body, current);
  }

  async function readStream(body, run) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let cut;
        while ((cut = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, cut).trim();
          buffer = buffer.slice(cut + 1);
          if (!line) continue;
          let message;
          try {
            message = JSON.parse(line);
          } catch (e) {
            continue; // a garbled line is skipped, not fatal
          }
          receive(message, run);
        }
      }
    } catch (e) {
      // fall through: a stream that ends without a result lost its connection
    }
    if (!run.gotResult) enqueue(run, { kind: "lost" });
  }

  function receive(message, run) {
    if (message.kind === "started") {
      run.id = message.run_id;
      $("run-title").textContent = message.title;
      run.clock = startClock();
      loadRuns();
    } else if (message.kind === "stage") {
      enqueue(run, message);
    } else if (message.kind === "result") {
      run.gotResult = true;
      enqueue(run, message);
    }
  }

  function enqueue(run, item) {
    run.queue.push(item);
    if (!run.pumping) pump(run);
  }

  async function pump(run) {
    run.pumping = true;
    while (run.queue.length) {
      const item = run.queue.shift();
      if (item.kind === "stage") await applyStage(run, item);
      else await finish(run, item);
    }
    run.pumping = false;
  }

  async function holdActive(run) {
    const shown = Date.now() - run.lastChange;
    if (run.active && shown < MIN_ACTIVE_MS) await sleep(MIN_ACTIVE_MS - shown);
  }

  async function applyStage(run, update) {
    const before = (run.stages[update.stage] || {}).status || "waiting";
    const ending = update.status === "done" || update.status === "failed";
    if (update.status === "active" && run.active !== update.stage) {
      await holdActive(run);
    } else if (ending && before === "waiting") {
      // a stage that finished in one step still gets its moment on screen
      await holdActive(run);
      drawStage(run, Object.assign({}, update, { status: "active", headline: "", note: null }));
      await sleep(MIN_ACTIVE_MS);
    } else if (ending) {
      await holdActive(run);
    }
    drawStage(run, update);
  }

  function drawStage(run, update) {
    run.stages[update.stage] = update;
    const card = $("stage-" + update.stage);
    if (!card) return;
    card.className = "stage " + update.status;
    card.querySelector(".mark").outerHTML = stageIcon(update.status);
    const head = card.querySelector(".head");
    head.textContent = update.status === "active" ? update.note || "In progress" : update.status === "waiting" ? "" : update.headline;

    const isNewActive = update.status === "active" && run.active !== update.stage;
    if (update.status === "active") {
      if (isNewActive) run.lastChange = Date.now();
      run.active = update.stage;
    }
    if (run.active === update.stage) {
      if (isNewActive || update.stage === "solving" || update.stage === "reading") {
        $("stage-panel").innerHTML = window.stageScene(update.stage, update.scene, update.facts);
      }
      $("explain-title").textContent = update.headline || LABEL[update.stage];
      $("explain-body").textContent = update.explanation || "";
      $("explain-note").textContent = update.note || "";
      $("explain-note").hidden = !update.note;
    }
  }

  async function finish(run, item) {
    await holdActive(run);
    clearInterval(run.clock);
    run.finished = true;
    if (item.kind === "lost") {
      outcome("warn", "The connection was lost", "If the analysis finished, it will appear under Past runs in a minute.");
      setTimeout(loadRuns, 30000);
      return;
    }
    const status = STATUS[item.status] || STATUS.error;
    if (run.active && !["passed", "failed"].includes(item.status)) {
      const active = run.stages[run.active];
      drawStage(run, Object.assign({}, active, { status: "failed", headline: status.badge }));
    }
    run.active = null;
    $("run-clock").textContent = "Finished.";
    await loadRuns();
    if (item.status === "passed" || item.status === "failed") {
      $("explain-title").textContent = status.title;
      $("explain-body").textContent = item.message;
      $("explain-note").hidden = true;
      await sleep(900);
      if (location.hash === "#/running") location.hash = "#/run/" + item.run_id;
      return;
    }
    outcome(status.tone, status.title, item.message, item.run_id);
  }

  function outcome(tone, title, message) {
    const box = $("run-outcome");
    box.className = "outcome tone-" + tone;
    box.innerHTML = "";
    const h = document.createElement("h2");
    h.textContent = title;
    const p = document.createElement("p");
    p.textContent = message;
    const again = document.createElement("a");
    again.className = "btn";
    again.href = "#/";
    again.textContent = "Try another brief";
    box.append(h, p, again);
    box.hidden = false;
  }

  // ----------------------------------------------------------- a report

  async function openRun(id) {
    show("report");
    const verdict = $("verdict");
    $("report").innerHTML = "";
    let data;
    try {
      const response = await fetch("/api/runs/" + id);
      if (!response.ok) throw new Error(response.status);
      data = await response.json();
    } catch (e) {
      $("report-title").textContent = "";
      $("report-when").textContent = "";
      verdict.className = "verdict tone-bad";
      verdict.innerHTML = "<h2>This run can't be found</h2><p>It may have been removed. Choose another from Past runs.</p>";
      $("dl-pdf").hidden = $("dl-md").hidden = true;
      $("report-label").hidden = true;
      $("report").hidden = true;
      return;
    }
    const meta = data.meta;
    const status = STATUS[meta.status] || STATUS.error;
    verdict.className = "verdict tone-" + status.tone;
    verdict.innerHTML = "";
    $("report-title").textContent = meta.title || "Your brief";
    $("report-when").textContent = when(meta.created) + (meta.duration_s ? ", took " + Math.round(meta.duration_s) + " s" : "");
    const h = document.createElement("h2");
    h.textContent = status.title;
    const p = document.createElement("p");
    p.textContent =
      meta.message ||
      (meta.status === "interrupted"
        ? "The page was closed or the connection dropped before it finished. Run the brief again to get a report."
        : meta.status === "running"
          ? "This analysis is still going. Check back in a minute."
          : "");
    verdict.append(h, p);

    $("dl-pdf").href = "/api/runs/" + id + "/report.pdf";
    $("dl-md").href = "/api/runs/" + id + "/report.md";
    $("dl-pdf").hidden = !meta.has_pdf;
    $("dl-md").hidden = !data.report_md;
    const article = $("report");
    article.hidden = false;
    if (data.report_md) {
      $("report-label").hidden = true;
      window.renderMarkdown(data.report_md, article, { foldCode: true });
    } else if (data.brief_md) {
      $("report-label").hidden = false;
      window.renderMarkdown(data.brief_md, article, { foldCode: true });
    } else {
      article.hidden = true;
    }
  }

  // ---------------------------------------------------------------- start

  wireUpload();
  window.addEventListener("hashchange", route);
  loadRuns();
  route();
})();

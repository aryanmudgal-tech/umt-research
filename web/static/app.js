// The professor's page: choose a brief, watch the run, read the report.
(function () {
  "use strict";

  const STAGES = [
    ["reading", "Reading the brief"],
    ["modeling", "Building the FE model"],
    ["solving", "Solving"],
    ["checking", "Verification checks"],
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

  const RUN_ROUTE = /^#\/run\/([0-9a-f-]+)(\/how)?$/;

  function route() {
    const hash = location.hash || "#/";
    const match = hash.match(RUN_ROUTE);
    markCurrent(match ? match[1] : null);
    closeMenus();
    if (match) {
      if (current && current.id === match[1] && !current.finished) {
        location.replace("#/running");
        return;
      }
      openRun(match[1], match[2] ? "how" : "report");
    } else if (hash === "#/running" && current) {
      show("run");
    } else {
      if (current && current.finished) {
        current = null;
        resetUpload();
      }
      $("still-running").hidden = !(current && !current.finished);
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

  function newBrief() {
    resetUpload();
    if (location.hash === "#/" || location.hash === "") route();
    else location.hash = "#/";
  }

  // ------------------------------------------------------------ past runs

  function when(iso) {
    const date = new Date(iso);
    if (isNaN(date)) return "";
    return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(date);
  }

  function el(tag, props, children) {
    const node = Object.assign(document.createElement(tag), props || {});
    (children || []).forEach((child) => node.append(child));
    return node;
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
    runs.forEach((run) => list.append(runItem(run)));
    const match = (location.hash || "").match(RUN_ROUTE);
    markCurrent(match ? match[1] : null);
  }

  function runItem(run) {
    const status = STATUS[run.status] || STATUS.error;
    const li = el("li");
    li.dataset.run = run.id;
    const live = current && current.id === run.id && !current.finished;
    const link = el("a", { href: live ? "#/running" : "#/run/" + run.id });
    link.dataset.run = run.id;
    link.append(
      el("span", { textContent: run.title || "Untitled brief" }),
      el("span", { className: "run-when" }, [
        el("span", { textContent: when(run.created) }),
        el("span", { className: "badge " + BADGE_CLASS[status.tone], textContent: status.badge }),
      ])
    );
    const more = el("button", { type: "button", className: "run-more", textContent: "⋯" });
    more.setAttribute("aria-label", "Options for " + (run.title || "this run"));
    more.setAttribute("aria-haspopup", "true");
    more.setAttribute("aria-expanded", "false");
    more.addEventListener("click", (e) => {
      e.stopPropagation();
      const open = more.getAttribute("aria-expanded") === "true";
      closeMenus();
      if (!open) openMenu(li, more, run);
    });
    li.append(el("div", { className: "row" }, [link, more]));
    return li;
  }

  function closeMenus() {
    document.querySelectorAll(".run-menu").forEach((menu) => menu.remove());
    document.querySelectorAll(".run-more[aria-expanded='true']").forEach((b) => b.setAttribute("aria-expanded", "false"));
  }

  function openMenu(li, button, run) {
    button.setAttribute("aria-expanded", "true");
    const rename = el("button", { type: "button", textContent: "Rename" });
    rename.addEventListener("click", () => startRename(li, run));
    const items = [rename];
    if (run.status !== "running") {
      const remove = el("button", { type: "button", className: "danger", textContent: "Delete" });
      remove.addEventListener("click", () => confirmDelete(li, run));
      items.push(remove);
    }
    const menu = el("div", { className: "run-menu" }, items);
    menu.setAttribute("role", "menu");
    items.forEach((item) => item.setAttribute("role", "menuitem"));
    li.append(menu);
    rename.focus();
  }

  function startRename(li, run) {
    closeMenus();
    const input = el("input", { type: "text", value: run.title || "", maxLength: 120 });
    input.setAttribute("aria-label", "New name for the run");
    const save = el("button", { type: "button", className: "btn btn-primary btn-small", textContent: "Save" });
    const cancel = el("button", { type: "button", className: "btn btn-small", textContent: "Cancel" });
    const form = el("div", { className: "rename" }, [input, el("div", { className: "actions" }, [save, cancel])]);
    li.replaceChildren(form);
    input.focus();
    input.select();
    const done = async (keep) => {
      const title = input.value.trim();
      if (keep && title && title !== run.title) {
        const response = await fetch("/api/runs/" + run.id, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title }),
        });
        if (response.ok && location.hash.match(RUN_ROUTE) && location.hash.includes(run.id)) {
          $("report-title").textContent = title;
        }
      }
      loadRuns();
    };
    save.addEventListener("click", () => done(true));
    cancel.addEventListener("click", () => done(false));
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") done(true);
      if (e.key === "Escape") done(false);
    });
  }

  function confirmDelete(li, run) {
    closeMenus();
    const yes = el("button", { type: "button", className: "btn btn-danger btn-small", textContent: "Delete" });
    const no = el("button", { type: "button", className: "btn btn-small", textContent: "Cancel" });
    li.replaceChildren(
      el("div", { className: "confirm" }, [
        el("p", { textContent: "Delete “" + (run.title || "this run") + "”? Its report and PDF go too." }),
        el("div", { className: "actions" }, [yes, no]),
      ])
    );
    no.focus();
    no.addEventListener("click", () => loadRuns());
    yes.addEventListener("click", async () => {
      await fetch("/api/runs/" + run.id, { method: "DELETE" });
      if ((location.hash || "").includes(run.id)) location.hash = "#/";
      loadRuns();
    });
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

  const ICON = {
    pass: '<svg class="icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8.5l3 3 7-7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    fail: '<svg class="icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
    skipped: '<svg class="icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M4 8h8" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
    fact: '<svg class="icon" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="2.5" fill="currentColor"/></svg>',
  };

  let shown = { id: null, data: null, story: null };

  async function openRun(id, tab) {
    show("report");
    if (shown.id !== id) {
      shown = { id, data: null, story: null };
      try {
        const response = await fetch("/api/runs/" + id);
        if (!response.ok) throw new Error(response.status);
        shown.data = await response.json();
      } catch (e) {
        missingRun();
        return;
      }
      fillReport(id, shown.data);
    }
    if (!shown.data.report_md && tab === "report") {
      location.replace("#/run/" + id + "/how"); // nothing to read but how it ended
      return;
    }
    $("tab-report").href = "#/run/" + id;
    $("tab-how").href = "#/run/" + id + "/how";
    $("tab-report").hidden = !shown.data.report_md;
    [["tab-report", "report"], ["tab-how", "how"]].forEach(([el_id, name]) => {
      if (tab === name) $(el_id).setAttribute("aria-current", "page");
      else $(el_id).removeAttribute("aria-current");
    });
    $("pane-report").hidden = tab !== "report";
    $("pane-how").hidden = tab !== "how";
    if (tab === "how") await showStory(id);
  }

  function missingRun() {
    $("report-title").textContent = "";
    $("report-when").textContent = "";
    const verdict = $("verdict");
    verdict.className = "verdict tone-bad";
    verdict.innerHTML = "<h2>This run can't be found</h2><p>It may have been deleted. Choose another from Past runs.</p>";
    $("dl-pdf").hidden = $("dl-md").hidden = true;
    document.querySelector(".tabs").hidden = true;
    $("pane-report").hidden = $("pane-how").hidden = true;
  }

  function fillReport(id, data) {
    document.querySelector(".tabs").hidden = false;
    const meta = data.meta;
    const status = STATUS[meta.status] || STATUS.error;
    const verdict = $("verdict");
    verdict.className = "verdict tone-" + status.tone;
    $("report-title").textContent = meta.title || "Your brief";
    $("report-when").textContent = when(meta.created) + (meta.duration_s ? ", took " + Math.round(meta.duration_s) + " s" : "");
    const message =
      meta.message ||
      (meta.status === "interrupted"
        ? "The page was closed or the connection dropped before it finished. Run the brief again to get a report."
        : meta.status === "running"
          ? "This analysis is still going. Check back in a minute."
          : "");
    verdict.replaceChildren(el("h2", { textContent: status.title }), el("p", { textContent: message }));

    $("dl-pdf").href = "/api/runs/" + id + "/report.pdf";
    $("dl-md").href = "/api/runs/" + id + "/report.md";
    $("dl-pdf").hidden = !meta.has_pdf;
    $("dl-md").hidden = !data.report_md;
    const article = $("report");
    if (data.report_md) {
      $("report-label").hidden = true;
      window.renderMarkdown(data.report_md, article, { foldCode: true });
    } else {
      article.innerHTML = "";
    }
    $("story").innerHTML = "";
  }

  async function showStory(id) {
    const list = $("story");
    if (shown.story) return;
    list.innerHTML = '<li class="muted">Loading the steps…</li>';
    try {
      const response = await fetch("/api/runs/" + id + "/story");
      if (!response.ok) throw new Error(response.status);
      shown.story = await response.json();
    } catch (e) {
      list.innerHTML = "";
      list.append(el("li", { className: "muted", textContent: "No record of the steps was kept for this run." }));
      return;
    }
    list.innerHTML = "";
    shown.story.steps.forEach((step) => list.append(storyStep(step)));
  }

  function storyStep(step) {
    const body = el("div", {}, [el("h3", { textContent: step.title }), el("p", { className: "summary", textContent: step.summary })]);
    if (step.tex) {
      const eq = el("div", { className: "equation" });
      try {
        window.katex.render(step.tex, eq, { displayMode: true, throwOnError: false });
      } catch (e) {
        eq.textContent = step.tex;
      }
      body.append(eq);
    }
    if (step.items && step.items.length) {
      const items = el("ul", { className: "items" });
      step.items.forEach((item) => items.append(storyItem(item)));
      body.append(items);
    }
    if (step.verdict) {
      body.append(el("p", { className: "verdict-line " + (step.status === "failed" ? "bad" : "ok"), textContent: step.verdict }));
    }
    // the step number is the li's ::before, which takes the grid's first column
    return el("li", { className: step.status === "failed" ? "failed" : "done" }, [body]);
  }

  function storyItem(item) {
    const kind = item.status || "fact";
    const li = el("li", { className: kind });
    li.innerHTML = ICON[kind] || ICON.fact;
    const text = el("div");
    const title = el("strong", { textContent: item.text });
    text.append(title);
    if (kind === "skipped") title.after(el("span", { className: "tag skip", textContent: "Skipped" }));
    if (kind === "fail") title.after(el("span", { className: "tag fail", textContent: "Failed" }));
    if (item.detail) text.append(el("p", { className: "what", textContent: item.detail }));
    if (item.evidence) text.append(el("p", { className: "evidence", textContent: item.evidence }));
    li.append(text);
    return li;
  }

  // ---------------------------------------------------------------- start

  wireUpload();
  $("new-brief").addEventListener("click", newBrief);
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".run-menu")) closeMenus();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeMenus();
  });
  window.addEventListener("hashchange", route);
  loadRuns();
  route();
})();

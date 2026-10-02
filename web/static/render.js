// Markdown to a formatted document, with typeset math. Shared by the page and
// the PDF printer so both show the same thing.
(function () {
  "use strict";

  // html: false escapes any raw HTML in a brief instead of running it.
  const md = window.markdownit({ html: false, linkify: true, typographer: false }).use(window.texmath, {
    engine: window.katex,
    delimiters: "dollars",
    katexOptions: { throwOnError: false },
  });

  // foldCode: tuck raw data blocks (the model JSON, the verdict) behind
  // "Show the data", so the professor reads results, not code.
  window.renderMarkdown = function (text, el, opts) {
    const options = opts || {};
    el.innerHTML = md.render(text || "");
    el.querySelectorAll("table").forEach(function (table) {
      const wrap = document.createElement("div");
      wrap.className = "table-wrap";
      table.parentNode.insertBefore(wrap, table);
      wrap.appendChild(table);
    });
    if (options.foldCode) {
      el.querySelectorAll("pre").forEach(function (pre) {
        const details = document.createElement("details");
        details.className = "data";
        const summary = document.createElement("summary");
        summary.textContent = "Show the data";
        pre.parentNode.insertBefore(details, pre);
        details.appendChild(summary);
        details.appendChild(pre);
      });
    }
  };
})();

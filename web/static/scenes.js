// One animated line drawing per stage, adapted to the brief: soil springs only
// with a soil term, end thrust only with an axial force, a taper only when the
// stiffness varies. window.stageScene(stage, scene, facts) returns SVG markup.
(function () {
  "use strict";

  const W = 640, H = 170, X0 = 60, X1 = 580, Y = 78;

  function esc(text) {
    return String(text).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  function svg(body, label) {
    return '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + esc(label) + '">' + body + "</svg>";
  }

  function supportKinds(facts) {
    const line = (facts || []).find(function (f) { return f.indexOf("Supports:") === 0; });
    const kinds = line ? line.slice(9).trim().split(" and ") : ["pin", "roller"];
    return [kinds[0] || "pin", kinds[kinds.length - 1] || "roller"];
  }

  // a support drawn at x under a beam whose underside is at y
  function support(kind, x, y) {
    const ground = '<line class="line" x1="' + (x - 18) + '" y1="' + (y + 26) + '" x2="' + (x + 18) + '" y2="' + (y + 26) + '"/>';
    let hatch = "";
    for (let k = -16; k <= 12; k += 7) {
      hatch += '<line class="line-faint" x1="' + (x + k) + '" y1="' + (y + 27) + '" x2="' + (x + k - 6) + '" y2="' + (y + 34) + '"/>';
    }
    if (kind === "fixed") {
      let wall = '<line class="line" x1="' + x + '" y1="' + (y - 26) + '" x2="' + x + '" y2="' + (y + 26) + '"/>';
      for (let k = -24; k <= 20; k += 8) {
        wall += '<line class="line-faint" x1="' + x + '" y1="' + (y + k) + '" x2="' + (x - 8) + '" y2="' + (y + k + 7) + '"/>';
      }
      return wall;
    }
    if (kind === "roller") {
      return '<circle class="line" cx="' + x + '" cy="' + (y + 16) + '" r="9"/>' + ground + hatch;
    }
    return '<polygon class="line" points="' + x + "," + (y + 4) + " " + (x - 12) + "," + (y + 26) + " " + (x + 12) + "," + (y + 26) + '"/>' + ground + hatch;
  }

  function supports(facts, y) {
    const kinds = supportKinds(facts);
    return support(kinds[0], X0, y) + support(kinds[1], X1, y);
  }

  // the beam's centreline with a sag of s at midspan, as a cubic Bezier
  function curve(s, dy) {
    const y = Y + (dy || 0);
    return "M" + X0 + " " + y + " C 160 " + (y + s) + ", 480 " + (y + s) + ", " + X1 + " " + y;
  }

  function sagAnimation(dy) {
    return '<animate attributeName="d" dur="2.4s" repeatCount="indefinite" values="' +
      curve(0, dy) + ";" + curve(22, dy) + ";" + curve(0, dy) + '"/>';
  }

  // a tapered beam: depth grows from the left end to the right end
  function taperPath(s) {
    const top = "M" + X0 + " " + (Y - 4) + " C 160 " + (Y - 6 + s) + ", 480 " + (Y - 10 + s) + ", " + X1 + " " + (Y - 11);
    const bottom = " L" + X1 + " " + (Y + 11) + " C 480 " + (Y + 10 + s) + ", 160 " + (Y + 6 + s) + ", " + X0 + " " + (Y + 4) + " Z";
    return top + bottom;
  }

  const scenes = {
    reading: function (scene, facts) {
      let body = '<rect class="doc-page" x="40" y="16" width="190" height="130" rx="6"/>' +
        '<rect class="scan a-scan" x="48" y="26" width="174" height="16" rx="3"/>';
      [150, 120, 140, 100, 145, 80].forEach(function (w, k) {
        body += '<rect class="doc-line" x="58" y="' + (32 + 16 * k) + '" width="' + w + '" height="5" rx="2"/>';
      });
      const shown = (facts && facts.length ? facts : ["Span", "Supports", "Load", "Equation"]).slice(0, 5);
      shown.forEach(function (fact, k) {
        const width = Math.min(330, 26 + fact.length * 7.4);
        body += '<g class="chip tb a-pop" style="--d:' + (0.25 + 0.45 * k) + 's;--dur:' + (1.2 + 0.45 * shown.length) + 's">' +
          '<rect x="270" y="' + (14 + 29 * k) + '" width="' + width + '" height="24" rx="6"/>' +
          '<text x="283" y="' + (31 + 29 * k) + '">' + esc(fact) + "</text></g>";
      });
      return svg(body, "The brief being read and its facts picked out");
    },

    modeling: function (scene, facts) {
      const n = Math.max(1, Math.min(scene.elements || 10, 24));
      let body = '<line class="beam a-draw" pathLength="100" x1="' + X0 + '" y1="' + Y + '" x2="' + X1 + '" y2="' + Y + '"/>';
      for (let k = 0; k <= n; k++) {
        const x = X0 + ((X1 - X0) * k) / n;
        body += '<circle class="node-dot tb a-pop" style="--d:' + (0.05 + (1.6 * k) / n).toFixed(2) + 's;--dur:3s" cx="' + x.toFixed(1) + '" cy="' + Y + '" r="4.5"/>';
      }
      body += supports(facts, Y);
      body += '<text class="svg-label a-fade" style="--dur:3s" x="320" y="150" text-anchor="middle">Beam split into ' +
        (scene.elements || n) + " short pieces</text>";
      return svg(body, "The beam being split into short pieces");
    },

    solving: function (scene, facts) {
      let body = "";
      // the load
      body += '<g class="load a-bob">';
      for (let x = 95; x <= 545; x += 64) {
        body += '<line x1="' + x + '" y1="12" x2="' + x + '" y2="40"/><polygon points="' + (x - 5) + ",38 " + (x + 5) + ",38 " + x + ',46"/>';
      }
      body += "</g>";
      if (scene.soil) {
        [130, 215, 300, 385, 470].forEach(function (x) {
          body += '<polyline class="line bt a-squash" points="' + x + ",132 " + x + ",122 " + (x - 8) + ",117 " + (x + 8) + ",110 " +
            (x - 8) + ",103 " + (x + 8) + ",96 " + x + ",91 " + x + ',86"/>';
        });
        body += '<line class="line-faint" x1="90" y1="133" x2="550" y2="133"/>';
      }
      if (scene.axial) {
        body += '<g class="load a-push" style="--dx:6px"><line x1="8" y1="' + Y + '" x2="40" y2="' + Y + '"/><polygon points="38,' + (Y - 5) + " 38," + (Y + 5) + " 47," + Y + '"/></g>';
        body += '<g class="load a-push" style="--dx:-6px"><line x1="632" y1="' + Y + '" x2="600" y2="' + Y + '"/><polygon points="602,' + (Y - 5) + " 602," + (Y + 5) + " 593," + Y + '"/></g>';
      }
      if (scene.tapered) {
        body += '<path class="beam-fill" d="' + taperPath(0) + '"><animate attributeName="d" dur="2.4s" repeatCount="indefinite" values="' +
          taperPath(0) + ";" + taperPath(22) + ";" + taperPath(0) + '"/></path>';
      } else {
        body += '<path class="beam" d="' + curve(0) + '">' + sagAnimation(0) + "</path>";
      }
      body += supports(facts, Y);
      let label = "The load pushes down and the beam bends";
      if (scene.soil) label = "The load pushes down and the soil pushes back";
      if (scene.axial) label = "The load bends the beam and the end thrust adds to it";
      if (scene.sag_mm) label = "Midspan sag " + Number(scene.sag_mm).toPrecision(3) + " mm";
      body += '<text class="svg-label" x="320" y="160" text-anchor="middle">' + esc(label) + "</text>";
      return svg(body, "The beam deflecting under its load");
    },

    checking: function () {
      const checks = ["Forces balance", "Supports stay put", "The answer satisfies the equations", "The method gets a known answer right"];
      let body = '<g transform="translate(150,22)">';
      checks.forEach(function (text, k) {
        const y = 32 * k;
        body += '<circle class="line-faint" cx="10" cy="' + (y + 10) + '" r="10"/>' +
          '<path class="svg-ok tb a-pop" style="--d:' + (0.3 + 0.6 * k) + 's" d="M4 ' + (y + 10) + " l4 4 l8 -8" + '"/>' +
          '<text class="svg-label-strong" x="32" y="' + (y + 15) + '">' + esc(text) + "</text>";
      });
      return svg(body + "</g>", "The checks being ticked off");
    },

    independent: function () {
      const d = "M70 42 C 150 120, 490 120, 570 42";
      const body = '<path class="line-faint" style="stroke-width:7" d="' + d + '"/>' +
        '<path class="svg-ok a-draw" pathLength="100" d="' + d + '"/>' +
        '<line class="line-faint" style="stroke-width:7" x1="408" y1="18" x2="432" y2="18"/><text class="svg-label" x="440" y="23">Our solver</text>' +
        '<line class="svg-ok" x1="408" y1="38" x2="432" y2="38"/><text class="svg-label" x="440" y="43">Independent method</text>' +
        '<text class="svg-label-strong a-fade" style="--dur:3s" x="320" y="152" text-anchor="middle">Comparing the two answers</text>';
      return svg(body, "Two independent answers being compared");
    },

    report: function () {
      let body = '<rect class="doc-page" x="200" y="12" width="240" height="146" rx="6"/>' +
        '<rect class="doc-head tl a-grow" x="218" y="26" width="120" height="9" rx="2"/>';
      [200, 170, 185].forEach(function (w, k) {
        body += '<rect class="doc-line tl a-grow" style="--d:' + (0.3 + 0.2 * k) + 's" x="218" y="' + (46 + 11 * k) + '" width="' + w + '" height="5" rx="2"/>';
      });
      body += '<g class="a-fade" style="--d:.5s"><rect class="line-faint" x="218" y="96" width="204" height="48" rx="2"/>' +
        '<line class="line-faint" x1="218" y1="112" x2="422" y2="112"/><line class="line-faint" x1="218" y1="128" x2="422" y2="128"/>' +
        '<line class="line-faint" x1="320" y1="96" x2="320" y2="144"/></g>';
      return svg(body, "The report being written");
    },
  };

  window.stageScene = function (stage, scene, facts) {
    const draw = scenes[stage];
    let markup = draw ? draw(scene || {}, facts || []) : "";
    if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      markup = markup.replace(/<animate [^>]*\/>/g, "");
    }
    return markup;
  };
})();

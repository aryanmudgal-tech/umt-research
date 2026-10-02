// Markdown to a formatted document, with typeset math. Shared by the page and
// the PDF printer so both show the same thing.
//
// A governing-equation spec (the JSON with "coeffs") is never shown as code:
// wherever it appears in a brief, fenced or pasted straight into the text, it
// becomes an equation card, the equation typeset with its parameters named in
// engineering units. The card is built from the JSON itself, so it shows
// exactly what the solver will be given.
(function () {
  "use strict";

  const EQUATION_FENCE = "beam-equation";

  // html: false escapes any raw HTML in a brief instead of running it.
  const md = window.markdownit({ html: false, linkify: true, typographer: false }).use(window.texmath, {
    engine: window.katex,
    delimiters: "dollars",
    katexOptions: { throwOnError: false },
  });

  function esc(text) {
    return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  }

  // ------------------------------------------------- finding equation specs

  function isSpec(value) {
    return value && typeof value === "object" && !Array.isArray(value) && value.coeffs && typeof value.coeffs === "object";
  }

  // the end of the balanced {...} starting at text[start], strings respected
  function objectEnd(text, start) {
    let depth = 0;
    let inString = false;
    for (let i = start; i < text.length; i++) {
      const c = text[i];
      if (inString) {
        if (c === "\\") i++;
        else if (c === '"') inString = false;
      } else if (c === '"') inString = true;
      else if (c === "{") depth++;
      else if (c === "}" && --depth === 0) return i;
    }
    return -1;
  }

  function fenced(spec) {
    return "\n\n```" + EQUATION_FENCE + "\n" + JSON.stringify(spec) + "\n```\n\n";
  }

  // Replace every equation spec in the markdown with a beam-equation fence.
  function liftSpecs(text) {
    // fenced json blocks first
    text = text.replace(/```(?:json)?[ \t]*\n([\s\S]*?)\n```/g, (whole, body) => {
      try {
        const value = JSON.parse(body);
        return isSpec(value) ? fenced(value) : whole;
      } catch (e) {
        return whole;
      }
    });
    // then bare objects pasted into the text, outside any fence
    let out = "";
    let i = 0;
    let inFence = false;
    while (i < text.length) {
      if (text.startsWith("```", i) && (i === 0 || text[i - 1] === "\n")) {
        inFence = !inFence;
        const eol = text.indexOf("\n", i);
        const next = eol === -1 ? text.length : eol + 1;
        out += text.slice(i, next);
        i = next;
        continue;
      }
      if (!inFence && text[i] === "{") {
        const end = objectEnd(text, i);
        if (end > i) {
          try {
            const value = JSON.parse(text.slice(i, end + 1));
            if (isSpec(value)) {
              out += fenced(value);
              i = end + 1;
              continue;
            }
          } catch (e) {
            // not JSON: leave it as written
          }
        }
      }
      out += text[i];
      i++;
    }
    return out;
  }

  // --------------------------------------------- the equation, typeset

  const FUNCTIONS = { sin: "\\sin", cos: "\\cos", tan: "\\tan", exp: "\\exp", log: "\\ln", sinh: "\\sinh", cosh: "\\cosh", tanh: "\\tanh" };

  function identTex(name) {
    if (name === "pi") return "\\pi";
    const m = name.match(/^([A-Za-z]+)(\d+)$/);
    if (m) return m[1] + "_{" + m[2] + "}";
    if (name.length > 1 && name.indexOf("_") > 0) {
      const [base, sub] = name.split("_", 2);
      return base + "_{\\mathrm{" + sub + "}}";
    }
    return name.length > 1 ? "\\mathrm{" + name + "}" : name;
  }

  // A coefficient such as "E*I0*(1 + x/L)**0.5" as TeX. Arithmetic only, as
  // the solver accepts; anything unexpected is shown literally.
  function exprTex(source) {
    const s = String(source).replace(/\s+/g, "");
    let i = 0;
    function powerOperand() {
      if (s[i] === "(") {
        i++;
        const inner = group(")");
        i++;
        return inner;
      }
      let token = "";
      while (i < s.length && /[A-Za-z0-9_.]/.test(s[i])) token += s[i++];
      return token;
    }
    function group(closing) {
      let out = "";
      while (i < s.length && s[i] !== closing) {
        const c = s[i];
        if (/[A-Za-z_]/.test(c)) {
          let name = "";
          while (i < s.length && /[A-Za-z0-9_]/.test(s[i])) name += s[i++];
          if (s[i] === "(") {
            i++;
            const inner = group(")");
            i++;
            if (name === "sqrt") out += "\\sqrt{" + inner + "}";
            else if (name === "Abs") out += "\\left|" + inner + "\\right|";
            else out += (FUNCTIONS[name] || "\\operatorname{" + name + "}") + "\\left(" + inner + "\\right)";
          } else {
            out += (out && /[A-Za-z0-9}]$/.test(out) ? " " : "") + identTex(name);
          }
        } else if (/[0-9.]/.test(c)) {
          let num = "";
          while (i < s.length && /[0-9.]/.test(s[i])) num += s[i++];
          if (/[eE]/.test(s[i] || "") && /[-+0-9]/.test(s[i + 1] || "")) {
            let exp = s[++i];
            i++;
            while (i < s.length && /[0-9]/.test(s[i])) exp += s[i++];
            out += num + " \\times 10^{" + Number(exp) + "}";
          } else {
            out += num;
          }
        } else if (c === "*" && s[i + 1] === "*") {
          i += 2;
          out += "^{" + powerOperand() + "}";
        } else if (c === "*") {
          i++; // implicit product, as an engineer writes EI
        } else if (c === "(") {
          i++;
          out += "\\left(" + group(")") + "\\right)";
          i++;
        } else {
          out += c === "-" ? " - " : c === "+" ? " + " : c === "/" ? " / " : esc(c);
          i++;
        }
      }
      return out;
    }
    try {
      return group(undefined);
    } catch (e) {
      return "\\text{" + esc(source) + "}";
    }
  }

  function isZero(text) {
    return text === undefined || text === null || /^\s*0*\.?0*\s*$/.test(String(text));
  }

  // a4 v'''' + a2 v'' + a1 v' + a0 v = f, with an x-dependent a4 or a2 written
  // in the self-adjoint form the solver uses: (EI(x) v'')''
  function equationTex(spec) {
    const c = spec.coeffs || {};
    const terms = [];
    const wrap = (tex) => (/[+-]/.test(tex) ? "\\left(" + tex + "\\right)" : tex);
    if (!isZero(c.v4)) {
      const a = exprTex(c.v4);
      terms.push(/x/.test(c.v4) ? "\\left(" + a + "\\, v''\\right)''" : wrap(a) + "\\, v''''");
    }
    if (!isZero(c.v2)) {
      const a = exprTex(c.v2);
      terms.push(/x/.test(c.v2) ? "\\left(" + a + "\\, v'\\right)'" : wrap(a) + "\\, v''");
    }
    if (!isZero(c.v1)) terms.push(wrap(exprTex(c.v1)) + "\\, v'");
    if (!isZero(c.v0)) terms.push(wrap(exprTex(c.v0)) + "\\, v");
    return (terms.join(" + ") || "0") + " = " + exprTex(isZero(spec.rhs) ? "0" : spec.rhs);
  }

  // ------------------------------------------ the parameters, in units

  const SUPERSCRIPT = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };

  function plain(value, unit) {
    const v = Number(value);
    if (v !== 0 && (Math.abs(v) >= 1e5 || Math.abs(v) < 1e-3)) {
      const exp = Math.floor(Math.log10(Math.abs(v)));
      const mant = Number((v / Math.pow(10, exp)).toPrecision(3));
      return mant + " × 10" + String(exp).split("").map((d) => SUPERSCRIPT[d]).join("") + (unit ? " " + unit : "");
    }
    return Number(v.toPrecision(4)) + (unit ? " " + unit : "");
  }

  function scaled(value, steps) {
    const v = Number(value);
    for (const [size, unit] of steps) if (Math.abs(v) >= size) return Number((v / size).toPrecision(4)) + " " + unit;
    return plain(v, steps[steps.length - 1][1]);
  }

  // what each parameter is to a civil engineer, judged by its name
  function describe(name, value) {
    const v = Number(value);
    const pressure = [[1e9, "GPa"], [1e6, "MPa"], [1e3, "kPa"], [1, "Pa"]];
    if (name === "E") return ["Elastic modulus", scaled(v, pressure)];
    if (name === "G") return ["Shear modulus", scaled(v, pressure)];
    if (name === "EI") return ["Flexural rigidity", plain(v, "N·m²")];
    if (/^I/.test(name)) return ["Second moment of area", plain(v, "m⁴")];
    if (name === "A") return ["Cross-sectional area", plain(v, "m²")];
    if (name === "k" || /^k_?s/.test(name)) return ["Modulus of subgrade reaction", plain(v, "N/m²")];
    if (/^(q|w)/.test(name)) {
      return ["Distributed load", Number((Math.abs(v) / 1000).toPrecision(4)) + " kN/m, " + (v < 0 ? "downward" : "upward")];
    }
    if (/^(P|N)/.test(name)) {
      return ["Axial force", scaled(Math.abs(v), [[1e6, "MN"], [1e3, "kN"], [1, "N"]]) + ", " + (v >= 0 ? "compression" : "tension")];
    }
    if (name === "L") return ["Span", plain(v, "m")];
    return ["", plain(v, "(SI units)")];
  }

  function equationCard(spec) {
    const title = spec.label ? "Governing equation: " + esc(spec.label) : "Governing equation";
    let tex;
    try {
      tex = window.katex.renderToString(equationTex(spec), { displayMode: true, throwOnError: false });
    } catch (e) {
      tex = "<code>" + esc(JSON.stringify(spec.coeffs)) + "</code>";
    }
    const rows = Object.entries(spec.params || {})
      .map(([name, value]) => {
        const [what, shown] = describe(name, value);
        const symbol = window.katex.renderToString(identTex(name), { throwOnError: false });
        return '<tr><td class="sym">' + symbol + "</td><td>" + esc(what) + "</td><td>" + esc(shown) + "</td></tr>";
      })
      .join("");
    return (
      '<figure class="eq-card">' +
      "<figcaption><strong>" + title + "</strong>" +
      (spec._comment ? "<span>" + esc(String(spec._comment).replace(/\*/g, "·")) + "</span>" : "") +
      '<span class="source">Read directly from the brief: this is the equation the solver is given.</span></figcaption>' +
      '<div class="eq">' + tex + "</div>" +
      (rows ? '<table class="eq-params"><tbody>' + rows + "</tbody></table>" : "") +
      "</figure>"
    );
  }

  const fence = md.renderer.rules.fence;
  md.renderer.rules.fence = function (tokens, idx, options, env, self) {
    const token = tokens[idx];
    if (token.info.trim() === EQUATION_FENCE) {
      try {
        return equationCard(JSON.parse(token.content));
      } catch (e) {
        // fall through to an ordinary code block
      }
    }
    return fence(tokens, idx, options, env, self);
  };

  // foldCode: tuck raw data blocks (the model JSON, the verdict) behind
  // "Show raw data", so the professor reads results, not code.
  window.renderMarkdown = function (text, el, opts) {
    const options = opts || {};
    el.innerHTML = md.render(liftSpecs(text || ""));
    el.querySelectorAll("table:not(.eq-params)").forEach(function (table) {
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
        summary.textContent = "Show raw data";
        pre.parentNode.insertBefore(details, pre);
        details.appendChild(summary);
        details.appendChild(pre);
      });
    }
  };

  window.equationTex = equationTex; // exposed for tests
})();

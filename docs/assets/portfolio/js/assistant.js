/* ==========================================================================
   Abdellatif Belmady — AI assistant (n8n workflow)
   - floating chat bubble: visitors ask about Abdellatif's background or the
     page they are reading; every question carries the page context
   - select any text → "Explain" or "Ask", the answer appears next to it
   Endpoint: config.extra.chatbot_url (zensical.toml), exposed by theme/main.html
   in <meta name="ab:chatbot">. To test another instance without rebuilding:
   localStorage.setItem("ab.chat.endpoint", "https://…/webhook/…/chat").
   Contract: POST { action, sessionId, chatInput, language, mode, metadata }
   → { output, suggestions }; an empty chatbot_url hides the assistant.
   ========================================================================== */
(function () {
  "use strict";

  /* ---------- helpers ---------- */
  var cfg = {};
  try { cfg = JSON.parse(document.getElementById("__config").textContent); } catch (e) { /* outside the theme */ }
  var base = new URL(cfg.base || ".", location.href).href;

  var STR = {
    "as.name": "Abdellatif's AI assistant", "as.tag": "Ask about Abdellatif's background or this page",
    "as.launch": "Ask my AI assistant", "as.open": "Open the AI assistant", "as.close": "Close the assistant", "as.clear": "New conversation",
    "as.placeholder": "Ask a question…", "as.send": "Send", "as.context": "Reading",
    "as.welcome": "Hi! 👋 I'm Abdellatif's AI assistant. Ask me about Abdellatif's **experience, skills or education**, or **select any text** on the site and I'll explain it.",
    "as.note": "AI answers can be wrong. For anything important, contact Abdellatif directly.",
    "as.error": "Sorry, I can't reach the assistant right now. Please try again in a moment.",
    "as.notconfigured": "The assistant isn't connected yet: set `chatbot_url` in `zensical.toml` (see `tools/README.md`).",
    "sel.explain": "Explain", "sel.ask": "Ask", "sel.explainTitle": "Explanation", "sel.askTitle": "Your question",
    "sel.placeholder": "Your question about this passage…", "sel.go": "Ask", "sel.loading": "Thinking…",
    "sel.continue": "Continue in the chat →", "sel.close": "Close",
    "ctx.home": "Home"
  };
  var STARTERS = {
    home: ["What does Abdellatif do at AXA?", "Summarize the career so far", "Which GenAI projects?", "How can I get in touch?"],
    experience: ["Summarize the experience", "Which projects used LLMs?", "What was built at Wineo?", "Computer vision experience?"],
    education: ["Where did Abdellatif study?", "Which certifications?", "Which languages does Abdellatif speak?"],
    blog: ["What is this blog about?", "Summarize the latest post"],
    post: ["Summarize this article", "Why does it matter for insurance?", "Explain the key idea simply"]
  };
  function t(k) { return STR[k] != null ? STR[k] : k; }
  function h(tag, attrs, children) {
    var el = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (v == null || v === false) return;
      if (k === "class") el.className = v;
      else if (k === "html") el.innerHTML = v;
      else if (k.indexOf("on") === 0 && typeof v === "function") el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    });
    (children == null ? [] : [].concat(children)).forEach(function (c) {
      if (c == null || c === false) return;
      el.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return el;
  }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function clip(s, n) { s = String(s || "").replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim(); return s.length > n ? s.slice(0, n) + "…" : s; }

  /* ---------- configuration ---------- */
  function endpoint() {
    var o = null;
    try { o = localStorage.getItem("ab.chat.endpoint"); } catch (e) { /* ignore */ }
    var m = document.querySelector('meta[name="ab:chatbot"]');
    var url = (o || (m && m.content) || "").trim();
    return /^https?:\/\//.test(url) && !/your-instance|<|>/.test(url) ? url : "";
  }
  var SID_KEY = "ab.chat.sid", HIST_KEY = "ab.chat.history";
  function uuid() {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
    return "ab-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 12);
  }
  function sessionId(reset) {
    var sid = null;
    try { sid = !reset && localStorage.getItem(SID_KEY); } catch (e) { /* ignore */ }
    if (!sid) { sid = uuid(); try { localStorage.setItem(SID_KEY, sid); } catch (e) { /* ignore */ } }
    return sid;
  }

  /* ---------- page context ---------- */
  var BLOCK = /^(P|DIV|LI|H[1-6]|UL|OL|TABLE|TR|BLOCKQUOTE|DT|DD|DL|FIGURE|FIGCAPTION|SECTION|ARTICLE|HEADER|ASIDE|PRE|DETAILS|SUMMARY)$/;
  function textOf(node) {
    var out = "";
    node.childNodes.forEach(function (ch) {
      if (ch.nodeType === 3) { out += ch.nodeValue; return; }
      if (ch.nodeType !== 1) return;
      var tag = ch.tagName;
      var head = /^H[1-4]$/.test(tag) ? "\n" + "#".repeat(+tag[1]) + " " : "";
      var block = BLOCK.test(tag);
      out += (head || (block ? "\n" : "")) + (tag === "LI" ? "- " : "") + textOf(ch) + (block || head ? "\n" : "");
    });
    return out;
  }
  function article() { return document.querySelector("article.md-content__inner"); }
  function zones() { return [document.querySelector(".ab-hero"), article()].filter(Boolean); }
  function pageText() {
    return clip(zones().map(function (z) {
      var c = z.cloneNode(true);
      c.querySelectorAll(".headerlink, script, style, button, svg, input, textarea, .ab-ask, .ab-seltool, .md-source-file, .md-feedback, #__comments, .giscus")
        .forEach(function (n) { n.remove(); });
      return textOf(c);
    }).join("\n"), 7000);
  }
  function rel() { return location.href.indexOf(base) === 0 ? location.href.slice(base.length).split("#")[0] : ""; }
  function kind() {
    var r = rel();
    if (!r || r === "index.html") return "home";
    if (/^experience\//.test(r)) return "experience";
    if (/^education\//.test(r)) return "education";
    if (/^blog\/?$/.test(r) || /^blog\/(page|archive|category)\//.test(r)) return "blog";
    if (/^blog\//.test(r)) return "post";
    return "other";
  }
  function pageInfo() {
    var art = article();
    var h1 = document.querySelector(".ab-hero__name") || (art && art.querySelector("h1"));
    var outline = art ? Array.prototype.map.call(art.querySelectorAll("h2"), function (x) { return x.textContent.replace("¶", "").trim(); }).join(" · ") : "";
    var title = kind() === "home" ? "Home" : h1 ? h1.textContent.replace("¶", "").trim() : document.title;
    return { url: location.pathname, path: rel(), kind: kind(), title: title, outline: clip(outline, 900), text: pageText() };
  }
  function contextLabel() { var i = pageInfo(); return i.kind === "home" ? t("ctx.home") : i.title; }

  /* ---------- n8n workflow call ---------- */
  function parseReply(raw) {
    var text = String(raw || "").trim();
    if (!text) return null;
    var data;
    try { data = JSON.parse(text); } catch (e) { return { reply: text }; }
    var item = Array.isArray(data) ? data[0] : data;
    if (typeof item === "string") return { reply: item };
    if (!item || typeof item !== "object") return null;
    var o = item.json && typeof item.json === "object" ? item.json : item;
    var reply = ["output", "text", "message", "response", "answer", "reply"].map(function (k) { return o[k]; })
      .filter(function (v) { return typeof v === "string" && v.trim(); })[0];
    if (!reply) return null;
    var sug = Array.isArray(o.suggestions) ? o.suggestions.filter(function (s) { return typeof s === "string" && s.trim() && s.length <= 80; }).slice(0, 3) : [];
    return { reply: reply.trim(), suggestions: sug };
  }
  function ask(mode, message, sel) {
    var url = endpoint();
    if (!url) return Promise.resolve({ reply: t("as.notconfigured"), suggestions: [], offline: true });
    var body = {
      action: "sendMessage", sessionId: sessionId(), chatInput: message || "", language: "en", mode: mode,
      metadata: {
        language: "en", mode: mode, source: "abdellatif-belmady-portfolio", timestamp: new Date().toISOString(),
        page: pageInfo(), selection: sel ? clip(sel.text, 1500) : "", passage: sel ? clip(sel.passage, 2000) : "",
        heading: sel ? sel.heading : ""
      }
    };
    var ctrl = window.AbortController ? new AbortController() : null;
    var timer = setTimeout(function () { if (ctrl) ctrl.abort(); }, 60000);
    return fetch(url, { method: "POST", headers: { "Content-Type": "application/json", Accept: "application/json, text/plain" },
      body: JSON.stringify(body), signal: ctrl ? ctrl.signal : undefined })
      .then(function (r) { return r.text().then(function (txt) { if (!r.ok) throw new Error("HTTP " + r.status); return txt; }); })
      .then(function (txt) { var p = parseReply(txt); if (!p) throw new Error("empty"); return p; })
      .finally(function () { clearTimeout(timer); });
  }

  /* ---------- small, safe Markdown for answers ---------- */
  function esc(s) { return String(s).replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); }
  function inline(s) {
    return esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>")
      .replace(/\[([^\]]+)\]\(((?:https?:\/\/|\/|mailto:)[^)\s]+)\)/g, function (m, label, href) {
        // site-relative links ("/experience/") resolve against the site root, not the domain root
        var url = href.charAt(0) === "/" ? new URL(href.slice(1), base).href : href;
        return '<a href="' + url + '">' + label + "</a>";
      });
  }
  function md(text) {
    var html = [], list = null, sub = false, code = null, table = null;
    function closeSub() { if (sub) { html.push("</ul></li>"); sub = false; } }
    function isRule(r) { return /^[\s|:-]+$/.test(r) && /-/.test(r); }
    function flushTable() {
      if (!table) return;
      var head = table.length > 1 && isRule(table[1]);
      var rows = table.filter(function (r) { return !isRule(r); }).map(function (r, i) {
        var tag = head && i === 0 ? "th" : "td";
        return "<tr>" + r.trim().replace(/^\||\|$/g, "").split("|").map(function (c) { return "<" + tag + ">" + inline(c.trim()) + "</" + tag + ">"; }).join("") + "</tr>";
      });
      html.push('<div class="ab-md-table"><table>' + rows.join("") + "</table></div>");
      table = null;
    }
    function close() { closeSub(); if (list) { html.push("</" + list + ">"); list = null; } }
    function append(s) { html[html.length - 1] = html[html.length - 1].replace(/<\/li>$/, "<br>" + s + "</li>"); }
    String(text || "").replace(/\r/g, "").split("\n").forEach(function (l) {
      var m, indent = /^\s{2,}\S/.test(l);
      if (code !== null) { if (/^\s*```/.test(l)) { html.push("<pre><code>" + esc(code.join("\n")) + "</code></pre>"); code = null; } else code.push(l); return; }
      if (/^\s*\|.*\|\s*$/.test(l)) { close(); (table = table || []).push(l); return; }
      if (l.trim()) flushTable();
      if (/^\s*```/.test(l)) { close(); code = []; return; }
      if (!l.trim() || /^\s*([-*_])(\s*\1){2,}\s*$/.test(l)) return;
      if (list && indent && (m = l.match(/^\s*[-*•]\s+(.*)/))) {
        if (!sub) { html[html.length - 1] = html[html.length - 1].replace(/<\/li>$/, ""); html.push("<ul>"); sub = true; }
        html.push("<li>" + inline(m[1]) + "</li>"); return;
      }
      if ((m = l.match(/^\s*[-*•]\s+(.*)/))) { if (list !== "ul") { close(); html.push("<ul>"); list = "ul"; } else closeSub(); html.push("<li>" + inline(m[1]) + "</li>"); return; }
      if ((m = l.match(/^\s*(\d+)[.)]\s+(.*)/))) {
        if (list !== "ol") { close(); html.push(+m[1] > 1 ? '<ol start="' + +m[1] + '">' : "<ol>"); list = "ol"; } else closeSub();
        html.push("<li>" + inline(m[2]) + "</li>"); return;
      }
      if (list && indent) { append(inline(l.trim())); return; }
      close();
      if (/^\s*#{1,6}\s+/.test(l)) { html.push("<p><strong>" + inline(l.replace(/^\s*#+\s+/, "")) + "</strong></p>"); return; }
      html.push("<p>" + inline(l) + "</p>");
    });
    flushTable();
    if (code) html.push("<pre><code>" + esc(code.join("\n")) + "</code></pre>");
    close();
    return html.join("");
  }

  var ICON = {
    chat: '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.4A8 8 0 1 1 21 12z"/><path d="M12 8.2l.9 2.3 2.3.9-2.3.9-.9 2.3-.9-2.3-2.3-.9 2.3-.9z" fill="currentColor" stroke="none"/></svg>',
    close: '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>',
    send: '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2 11 13"/><path d="M22 2 15 22l-4-9-9-4z"/></svg>',
    reset: '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/></svg>',
    explain: '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18h6"/><path d="M10 22h4"/><path d="M15.1 14c.2-1 .7-1.7 1.5-2.5A6 6 0 1 0 6 8c0 1.3.4 2.6 1.4 3.5.8.8 1.3 1.5 1.5 2.5"/></svg>',
    ask: '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3"/><path d="M12 17h.01"/></svg>'
  };

  /* ======================================================================
     CHAT BUBBLE (kept across instant navigation)
     ====================================================================== */
  var chat = {
    built: false, busy: false, history: [],
    build: function () {
      if (this.built && document.body.contains(this.root)) return;
      var self = this;
      try { this.history = JSON.parse(localStorage.getItem(HIST_KEY)) || []; } catch (e) { this.history = []; }
      this.launcher = h("button", { type: "button", class: "ab-asst-launcher", "aria-label": t("as.open"), title: t("as.open"), html: ICON.chat + "<span>" + esc(t("as.launch")) + "</span>" });
      this.ctx = h("span", { class: "ab-asst__ctx-text" });
      this.list = h("div", { class: "ab-asst__list", "aria-live": "polite" });
      this.sugBox = h("div", { class: "ab-asst__sugs" });
      this.input = h("textarea", { class: "ab-asst__input", rows: "1", placeholder: t("as.placeholder"), "aria-label": t("as.placeholder") });
      this.sendBtn = h("button", { type: "button", class: "ab-asst__send", "aria-label": t("as.send"), title: t("as.send"), html: ICON.send });
      this.panel = h("section", { class: "ab-asst", role: "dialog", "aria-label": t("as.name"), hidden: true }, [
        h("header", { class: "ab-asst__head" }, [
          h("span", { class: "ab-asst__avatar", "aria-hidden": "true" }),
          h("div", { class: "ab-asst__titles" }, [h("strong", null, t("as.name")), h("span", null, t("as.tag"))]),
          h("button", { type: "button", class: "ab-asst__icon", title: t("as.clear"), "aria-label": t("as.clear"), html: ICON.reset, onclick: function () { self.reset(); } }),
          h("button", { type: "button", class: "ab-asst__icon", title: t("as.close"), "aria-label": t("as.close"), html: ICON.close, onclick: function () { self.close(); } })
        ]),
        h("div", { class: "ab-asst__ctx" }, [h("span", { class: "ab-asst__ctx-label" }, t("as.context") + ":"), this.ctx]),
        this.list, this.sugBox,
        h("div", { class: "ab-asst__form" }, [this.input, this.sendBtn]),
        h("p", { class: "ab-asst__note" }, t("as.note"))
      ]);
      this.root = h("div", { class: "ab-asst-root" }, [this.panel, this.launcher]);
      document.body.appendChild(this.root);
      this.launcher.addEventListener("click", function () { if (self.panel.hidden) self.open(); else self.close(); });
      this.sendBtn.addEventListener("click", function () { self.send(); });
      this.input.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); self.send(); }
        if (e.key === "Escape") self.close();
      });
      this.input.addEventListener("input", function () { self.input.style.height = "auto"; self.input.style.height = Math.min(self.input.scrollHeight, 120) + "px"; });
      this.built = true;
      this.render();
    },
    open: function () {
      this.build();
      this.panel.hidden = false;
      this.root.classList.add("is-open");
      this.refresh();
      this.list.scrollTop = this.list.scrollHeight;
      this.input.focus({ preventScroll: true });
    },
    close: function () { if (!this.built) return; this.panel.hidden = true; this.root.classList.remove("is-open"); },
    save: function () { try { localStorage.setItem(HIST_KEY, JSON.stringify(this.history.slice(-30))); } catch (e) { /* ignore */ } },
    reset: function () { this.history = []; this.lastSuggestions = []; this.save(); sessionId(true); this.render(); },
    bubble: function (role, text, cls) {
      return h("div", { class: "ab-asst__msg ab-asst__msg--" + role + (cls ? " " + cls : "") }, [h("div", { class: "ab-asst__bubble", html: role === "user" ? esc(text).replace(/\n/g, "<br>") : md(text) })]);
    },
    render: function () {
      var self = this;
      this.list.textContent = "";
      this.list.appendChild(this.bubble("bot", t("as.welcome"), "is-welcome"));
      this.history.forEach(function (m) { self.list.appendChild(self.bubble(m.role, m.text, m.quote ? "has-quote" : null)); });
      this.suggest(this.history.length ? this.lastSuggestions || [] : this.starters());
      this.list.scrollTop = this.list.scrollHeight;
    },
    starters: function () { return STARTERS[kind()] || STARTERS.home; },
    suggest: function (list) {
      var self = this;
      this.sugBox.textContent = "";
      (list || []).forEach(function (s) {
        self.sugBox.appendChild(h("button", { type: "button", class: "ab-asst__sug", onclick: function () { self.send(s); } }, s));
      });
    },
    refresh: function () { if (this.built) { this.ctx.textContent = contextLabel(); if (!this.history.length) this.suggest(this.starters()); } },
    push: function (role, text, extra) {
      var m = { role: role, text: text };
      if (extra && extra.quote) m.quote = true;
      this.history.push(m); this.save();
      this.list.appendChild(this.bubble(role, text, m.quote ? "has-quote" : null));
      this.list.scrollTop = this.list.scrollHeight;
    },
    send: function (preset) {
      var self = this;
      var text = (preset != null ? preset : this.input.value).trim();
      if (!text || this.busy) return;
      this.build();
      if (this.panel.hidden) this.open();
      if (preset == null) { this.input.value = ""; this.input.style.height = "auto"; }
      this.push("user", text);
      this.suggest([]);
      this.busy = true;
      var typing = h("div", { class: "ab-asst__msg ab-asst__msg--bot" }, [h("div", { class: "ab-asst__bubble ab-asst__typing" }, [h("i"), h("i"), h("i")])]);
      this.list.appendChild(typing);
      this.list.scrollTop = this.list.scrollHeight;
      ask("chat", text).then(function (r) {
        typing.remove();
        self.push("bot", r.reply);
        self.lastSuggestions = r.suggestions || [];
        self.suggest(self.lastSuggestions);
      }, function () {
        typing.remove();
        self.list.appendChild(self.bubble("bot", t("as.error"), "is-error"));
      }).then(function () { self.busy = false; self.list.scrollTop = self.list.scrollHeight; });
    },
    /* continue in the chat an exchange that started from a selection */
    adopt: function (userText, reply) {
      this.build();
      this.push("user", userText, { quote: true });
      this.push("bot", reply);
      this.lastSuggestions = [];
      this.suggest([]);
      this.open();
    }
  };

  /* ======================================================================
     TEXT SELECTION → "Explain" / "Ask"
     ====================================================================== */
  var tool = null, pop = null, current = null;
  var hl = window.CSS && CSS.highlights && window.Highlight ? CSS.highlights : null;
  function inZone(node) { return zones().some(function (z) { return z.contains(node); }); }
  function blockOf(node) {
    var el = node.nodeType === 1 ? node : node.parentElement;
    return el ? el.closest("p, li, blockquote, td, th, dd, dt, figcaption, .admonition, h1, h2, h3, h4, .ab-card, .ab-highlight, .ab-job") : null;
  }
  function headingBefore(el) {
    var art = article();
    for (var n = el; n && n !== art; n = n.parentElement) {
      if (n.matches && n.matches(".ab-job, .ab-card")) { var hh = n.querySelector("h2, h3"); if (hh) return hh.textContent.replace("¶", "").trim(); }
      for (var s = n.previousElementSibling; s; s = s.previousElementSibling) {
        if (/^H[2-3]$/.test(s.tagName)) return s.textContent.replace("¶", "").trim();
      }
    }
    var h1 = art && art.querySelector("h1");
    return h1 ? h1.textContent.replace("¶", "").trim() : "";
  }
  function hideTool() { if (tool) tool.hidden = true; }
  function place(el, rect, below) {
    var w = el.offsetWidth, vw = document.documentElement.clientWidth;
    var left = clamp(rect.left + rect.width / 2 - w / 2 + window.scrollX, 8 + window.scrollX, vw - w - 8 + window.scrollX);
    var top = below ? rect.bottom + window.scrollY + 10 : rect.top + window.scrollY - el.offsetHeight - 10;
    if (!below && top < window.scrollY + 60) top = rect.bottom + window.scrollY + 10;
    el.style.left = left + "px"; el.style.top = top + "px";
  }
  function showTool(range, text) {
    if (!tool) {
      tool = h("div", { class: "ab-seltool", role: "toolbar", hidden: true }, [
        h("button", { type: "button", class: "ab-seltool__btn", "data-mode": "explain", html: ICON.explain + esc(t("sel.explain")) }),
        h("button", { type: "button", class: "ab-seltool__btn", "data-mode": "question", html: ICON.ask + esc(t("sel.ask")) })
      ]);
      /* keep the selection alive when the toolbar is clicked */
      tool.addEventListener("mousedown", function (e) { e.preventDefault(); });
      tool.addEventListener("click", function (e) {
        var b = e.target.closest(".ab-seltool__btn");
        if (b && current) openPop(b.getAttribute("data-mode"));
      });
      document.body.appendChild(tool);
    }
    var block = blockOf(range.startContainer);
    current = {
      range: range.cloneRange(), text: text,
      passage: block ? clip(block.textContent.replace("¶", ""), 2000) : "",
      heading: block ? headingBefore(block) : ""
    };
    tool.hidden = false;
    place(tool, range.getBoundingClientRect(), false);
  }
  function onSelection(e) {
    if (!endpoint()) return;
    if (e && e.target && e.target.closest && e.target.closest(".ab-seltool, .ab-ask, .ab-asst-root")) return;
    setTimeout(function () {
      var sel = window.getSelection();
      if (!sel || sel.isCollapsed || !sel.rangeCount) { hideTool(); return; }
      var text = sel.toString().replace(/\s+/g, " ").trim();
      var range = sel.getRangeAt(0);
      var anc = range.commonAncestorContainer.nodeType === 1 ? range.commonAncestorContainer : range.commonAncestorContainer.parentElement;
      if (!anc || !inZone(anc) || text.length < 3 || text.length > 2000 || anc.closest("textarea, input, .ab-ask")) { hideTool(); return; }
      showTool(range, text);
    }, 10);
  }
  function closePop() {
    if (pop) { pop.remove(); pop = null; }
    if (hl) hl.delete("ab-selection");
  }
  function openPop(mode) {
    var sel = current;
    closePop(); hideTool();
    if (hl) { try { hl.set("ab-selection", new Highlight(sel.range)); } catch (e) { /* ignore */ } }
    var answer = h("div", { class: "ab-ask__answer", "aria-live": "polite" });
    var foot = h("div", { class: "ab-ask__foot", hidden: true });
    var body = [h("blockquote", { class: "ab-ask__quote" }, clip(sel.text, 260))];
    var field = null, go = null;
    if (mode === "question") {
      field = h("textarea", { class: "ab-ask__input", rows: "2", placeholder: t("sel.placeholder"), "aria-label": t("sel.placeholder") });
      go = h("button", { type: "button", class: "ab-ask__go" }, t("sel.go"));
      body.push(h("div", { class: "ab-ask__form" }, [field, go]));
      var submit = function () {
        var q = field.value.trim();
        if (!q) { field.focus(); return; }
        go.disabled = field.disabled = true;
        run(q);
      };
      go.addEventListener("click", submit);
      field.addEventListener("keydown", function (e) { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } });
    }
    body.push(answer, foot);
    var title = mode === "explain" ? t("sel.explainTitle") : t("sel.askTitle");
    pop = h("div", { class: "ab-ask ab-ask--" + mode, role: "dialog", "aria-label": title }, [
      h("div", { class: "ab-ask__head" }, [
        h("span", { class: "ab-ask__title", html: (mode === "explain" ? ICON.explain : ICON.ask) + esc(title) }),
        /* removed after the click has finished bubbling, so no other handler sees a detached target */
        h("button", { type: "button", class: "ab-ask__close", "aria-label": t("sel.close"), title: t("sel.close"), html: ICON.close, onclick: function (e) { e.preventDefault(); setTimeout(closePop, 0); } })
      ])
    ].concat(body));
    document.body.appendChild(pop);
    place(pop, sel.range.getBoundingClientRect(), true);
    pop._rect = function () { return sel.range.getBoundingClientRect(); };
    window.getSelection().removeAllRanges();

    function run(question) {
      answer.innerHTML = '<p class="ab-ask__loading"><span class="ab-asst__typing"><i></i><i></i><i></i></span> ' + esc(t("sel.loading")) + "</p>";
      ask(mode, question || "", sel).then(function (r) {
        answer.innerHTML = md(r.reply);
        foot.textContent = "";
        if (!r.offline) {
          var userText = (mode === "explain" ? t("sel.explain") : question) + "\n“" + clip(sel.text, 200) + "”";
          foot.appendChild(h("button", { type: "button", class: "ab-ask__more", onclick: function () { chat.adopt(userText, r.reply); setTimeout(closePop, 0); } }, t("sel.continue")));
          foot.hidden = false;
        }
      }, function () {
        answer.innerHTML = md(t("as.error"));
        if (field) { field.disabled = false; go.disabled = false; }
      }).then(function () { if (pop) place(pop, pop._rect(), true); });
    }
    if (mode === "explain") run("");
    else field.focus({ preventScroll: true });
  }

  document.addEventListener("mouseup", onSelection);
  document.addEventListener("keyup", function (e) { if (e.shiftKey || e.key === "Shift") onSelection(e); });
  document.addEventListener("touchend", function (e) { setTimeout(function () { onSelection(e); }, 250); });
  document.addEventListener("keydown", function (e) {
    if (e.key !== "Escape") return;
    if (pop || (tool && !tool.hidden)) { closePop(); hideTool(); } else chat.close();
  });
  /* open chat: a click elsewhere closes it, except on the selection toolbar, the answer bubble or at the end of a text selection */
  document.addEventListener("click", function (e) {
    if (!chat.built || chat.panel.hidden) return;
    var path = e.composedPath ? e.composedPath() : [e.target];
    var inside = path.some(function (n) {
      return n === chat.root || (n.classList && (n.classList.contains("ab-ask") || n.classList.contains("ab-seltool")));
    });
    if (inside) return;
    var s = window.getSelection();
    if (s && !s.isCollapsed && String(s).trim()) return;
    chat.close();
  });
  document.addEventListener("mousedown", function (e) {
    if (pop && !e.target.closest(".ab-ask")) closePop();
    if (tool && !e.target.closest(".ab-seltool")) hideTool();
  });
  window.addEventListener("resize", function () { if (pop && pop._rect) place(pop, pop._rect(), true); hideTool(); });

  /* every page: fresh context, selection tools closed */
  function pageReady() { closePop(); hideTool(); if (!endpoint()) return; chat.build(); chat.refresh(); }
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(function () { setTimeout(pageReady, 0); });
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", pageReady);
  } else {
    setTimeout(pageReady, 0);
  }
  window.ABAssistant = { chat: chat, ask: ask, pageInfo: pageInfo };
})();

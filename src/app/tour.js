/* Guided tour for the clinician dashboard.
 *
 * Runs in the Streamlit page itself (injected once by src/app/tour.py). It switches tabs, presses the
 * replay's Play button, starts the what-if twin, spotlights one feature at a time and shows a caption,
 * advancing by itself. Space pauses, the arrow keys step, Esc exits. It only clicks the dashboard's own
 * controls, so it shows exactly what a user would see.
 */
(function () {
  "use strict";
  var W = window, D = document;
  if (W.__gdtTour) return;

  var C = { ink: "#0f1b2a", brand: "#0f766e", brandSoft: "#5eead4", card: "#ffffff", line: "#d9e2ea" };
  var FONT = "Geist, Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";

  // ------------------------------------------------------------------ helpers
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  function visible(el) { if (!el) return false; var r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; }
  function all(sel, root) { return Array.prototype.slice.call((root || D).querySelectorAll(sel)).filter(visible); }
  function byText(sel, re, root) { return all(sel, root).filter(function (el) { return re.test(el.textContent.trim()); })[0] || null; }
  function box(el) { return el && (el.closest('[data-testid="stElementContainer"]') || el.closest(".element-container") || el); }
  function next(el, n) {                       // el's element container and the n containers after it
    var out = [box(el)], cur = box(el);
    for (var i = 0; i < n && cur; i++) { cur = cur.nextElementSibling; if (cur && visible(cur)) out.push(cur); }
    return out;
  }
  function panel() { return all('[role="tabpanel"]')[0] || null; }
  function tabButton(name) { return byText('button[role="tab"]', new RegExp("^" + name)); }
  function section(title, root) {
    var t = byText(".gt-section .t", new RegExp("^" + title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")), root || panel() || D);
    return t ? t.closest(".gt-section") : null;
  }
  function chart(root, i) { return all(".js-plotly-plot", root || panel() || D)[i || 0] || null; }
  function sidebarWidget(label) {
    var sb = D.querySelector('[data-testid="stSidebar"]');
    var l = sb && byText('[data-testid="stWidgetLabel"]', new RegExp("^" + label), sb);
    return l ? box(l) : null;
  }
  async function waitFor(fn, ms) {
    var t = Date.now();
    while (Date.now() - t < (ms || 15000)) { var v = fn(); if (v && (!Array.isArray(v) || v.length)) return v; await sleep(150); }
    return null;
  }
  async function openTab(name) {
    var b = await waitFor(function () { return tabButton(name); }, 20000);
    if (b && b.getAttribute("aria-selected") !== "true") { b.click(); await sleep(450); }
  }
  function click(el) { if (el) el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, view: W })); }
  function openSidebar() {
    var sb = D.querySelector('[data-testid="stSidebar"]');
    if (sb && sb.getAttribute("aria-expanded") === "false") {
      var b = D.querySelector('[data-testid="stSidebarCollapsedControl"] button');
      if (b) b.click();
    }
  }

  // ------------------------------------------------------------------ the tour
  // dur: seconds on screen before moving on. target: element(s) to spotlight (null = no spotlight).
  var STEPS = [
    { title: "Guided tour", dur: 10, target: null,
      text: "A 2-hour warning for highs and lows in Type 2 diabetes" },
    { title: "Patient and model", dur: 22,
      before: async function () { openSidebar(); await sleep(300); },
      target: function () { return D.querySelector('[data-testid="stSidebar"]'); },
      text: "Held-out test patients · starts at the highest-risk moment" },
    { title: "Patient snapshot", dur: 20,
      before: async function () { await openTab("Live replay"); },
      target: function () { return all(".gt-hero")[0]; },
      text: "Health record and the time of the prediction" },
    { title: "Two risks", dur: 30,
      target: function () { return all(".gt-risk"); },
      text: "Low and high in the next 2 h · marker = alert threshold" },
    { title: "Live replay", dur: 12,
      before: async function () { await openTab("Live replay"); await waitFor(function () { return chart(); }); },
      target: function () { return box(chart()); },
      text: "The record plays forward; the model sees only the past" },
    { title: "The alert fires", dur: 30,
      before: async function () {
        var b = await waitFor(function () { var c = chart(); return c && c.querySelector(".updatemenu-button"); }, 8000);
        click(b);
      },
      until: function () {                      // stay until the replay has played to its last frame
        var c = chart(), s = c && c._fullLayout && c._fullLayout.sliders && c._fullLayout.sliders[0];
        return !s || s.active >= s.steps.length - 1;
      },
      target: function () { return box(chart()); },
      text: "Before the low, with its top reasons" },
    { title: "Every alert checked", dur: 20,
      target: function () { var s = section("What happens in this replay"); return s ? next(s, 1) : null; },
      text: "Warning time, false alarms and missed lows" },
    { title: "Glucose timeline", dur: 35,
      before: async function () { await openTab("Glucose timeline"); await waitFor(function () { return chart(); }); },
      target: function () { return box(chart()); },
      text: "Highs, lows, alerts, and what really happened" },
    { title: "Why the alert fired", dur: 35,
      before: async function () { await openTab("Alert drivers"); await waitFor(function () { return chart(); }); },
      target: function () { return box(chart()); },
      text: "SHAP: how much each input pushed the risk" },
    { title: "Reasons in plain words", dur: 25,
      target: function () { var c = box(chart()); return c ? next(c, 2) : null; },
      text: "Top reasons, and each data stream's share" },
    { title: "What-if twin", dur: 20,
      before: async function () {
        await openTab("What-if");
        var start = await waitFor(function () { return byText("button", /^Start the what-if twin/, panel()); }, 3000);
        if (start) { start.click(); }
        var opt = await waitFor(function () { return byText('label[data-baseweb="radio"]', /Extra insulin dose/, panel()); }, 30000);
        if (opt) { opt.click(); }
        await waitFor(function () { return all(".gt-compare", panel()); }, 30000);
      },
      target: function () { var l = byText('label[data-baseweb="radio"]', /Extra insulin dose/, panel()); return l ? next(l.closest('[data-testid="stRadio"]') || l, 1) : null; },
      text: "Trying an extra insulin dose" },
    { title: "Risk before → after", dur: 25,
      before: async function () { await openTab("What-if"); },
      target: function () { return all(".gt-compare", panel()); },
      text: "The risk model's view of the change" },
    { title: "Next 2 hours, simulated", dur: 40,
      target: function () { var c = chart(); return c ? next(c, 1) : null; },
      text: "As logged vs with the dose · 80% band" },
    { title: "Fitted to this patient", dur: 20,
      target: function () { var t = byText(".gt-chiprow", /Fit error/, panel()); return t ? next(t, 1) : null; },
      text: "Parameters from the first week of data" },
    { title: "Patient overview", dur: 20,
      before: async function () { await openTab("Patient overview"); await waitFor(function () { return section("Profile"); }); },
      target: function () { var s = section("Profile"); return s ? next(s, 3) : null; },
      text: "Medicines, conditions and latest labs" },
    { title: "14-day glucose summary", dur: 30,
      target: function () { var s = section("CGM summary"); return s ? next(s, 2) : null; },
      text: "Against international CGM targets" },
    { title: "Alerts over the record", dur: 20,
      target: function () { var s = section("Alerts over the record"); return s ? next(s, 1) : null; },
      text: "Alerts per day, one per risk episode" },
    { title: "Real-patient model", dur: 25,
      before: async function () { openSidebar(); await sleep(300); },
      target: function () { return sidebarWidget("Risk model"); },
      text: "Trained on 100 real patients (ShanghaiT2DM)" },
    { title: "That's the twin", dur: 12, target: null,
      text: "Warns early · explains itself · honest about what it knows" },
  ];

  // ------------------------------------------------------------------ overlay
  var css = D.createElement("style");
  css.textContent = [
    ".gdt-launch{position:fixed;right:22px;bottom:22px;z-index:999990;display:flex;align-items:center;gap:8px;",
    "padding:11px 18px;border:none;border-radius:999px;background:" + C.brand + ";color:#fff;font:600 14px/1 " + FONT + ";",
    "box-shadow:0 8px 24px -8px rgba(15,118,110,.7),0 2px 6px rgba(15,27,42,.18);cursor:pointer}",
    ".gdt-launch:hover{filter:brightness(1.08)}",
    ".gdt-spot{position:fixed;z-index:999991;border-radius:16px;pointer-events:none;",
    "box-shadow:0 0 0 3px " + C.brandSoft + ",0 0 0 9999px rgba(10,22,36,.58);transition:all .45s cubic-bezier(.2,.8,.2,1)}",
    ".gdt-dim{position:fixed;inset:0;z-index:999991;background:rgba(10,22,36,.58);pointer-events:none}",
    ".gdt-card{position:fixed;left:50%;transform:translateX(-50%);z-index:999992;width:max-content;max-width:calc(100vw - 32px);",
    "background:" + C.ink + ";color:#fff;border-radius:14px;padding:9px 10px 9px 16px;font-family:" + FONT + ";overflow:hidden;",
    "display:flex;align-items:center;gap:16px;box-shadow:0 12px 32px -10px rgba(0,0,0,.55);transition:top .35s ease,bottom .35s ease}",
    ".gdt-card .gdt-txt{min-width:0}",
    ".gdt-card .gdt-t{font-size:15px;font-weight:700;letter-spacing:-.005em;white-space:nowrap}",
    ".gdt-card .gdt-x{font-size:13px;line-height:1.35;color:#c9d6e0;margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}",
    ".gdt-card .gdt-k{font-size:11px;font-weight:600;color:" + C.brandSoft + ";font-variant-numeric:tabular-nums;white-space:nowrap}",
    ".gdt-card .gdt-ctl{display:flex;align-items:center;gap:4px;flex:none}",
    ".gdt-card .gdt-bar{position:absolute;left:0;right:0;bottom:0;height:3px;background:rgba(255,255,255,.12)}",
    ".gdt-card .gdt-fill{height:100%;width:0;background:" + C.brandSoft + "}",
    ".gdt-card button{background:rgba(255,255,255,.08);color:#fff;border:1px solid rgba(255,255,255,.16);border-radius:8px;",
    "width:28px;height:26px;padding:0;font:600 11px/1 " + FONT + ";cursor:pointer}",
    ".gdt-card button:hover{background:rgba(255,255,255,.2)}",
  ].join("");
  D.head.appendChild(css);

  var launch = D.createElement("button");
  launch.className = "gdt-launch";
  launch.innerHTML = "&#9654;&nbsp; Guided tour";
  launch.title = "Play an automatic tour of the dashboard";
  D.body.appendChild(launch);

  var spot = null, card = null, raf = 0, timer = 0;
  var S = { i: -1, paused: false, left: 0, started: 0, running: false, token: 0 };

  function el(tag, cls, html) { var e = D.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }

  function buildCard() {
    card = el("div", "gdt-card");
    card.innerHTML =
      '<div class="gdt-txt"><div class="gdt-t"></div><div class="gdt-x"></div></div>' +
      '<div class="gdt-ctl"><span class="gdt-k"></span>' +
      '<button data-a="prev" title="Previous (←)">&#9664;</button>' +
      '<button data-a="pause" title="Pause / resume (Space)">&#10073;&#10073;</button>' +
      '<button data-a="next" title="Next (→)">&#9654;</button>' +
      '<button data-a="stop" title="Exit (Esc)">&#10005;</button></div>' +
      '<div class="gdt-bar"><div class="gdt-fill"></div></div>';
    card.addEventListener("click", function (e) {
      var a = e.target.closest("button") && e.target.closest("button").getAttribute("data-a");
      if (a === "prev") go(S.i - 1); else if (a === "next") go(S.i + 1);
      else if (a === "pause") togglePause(); else if (a === "stop") stop();
    });
    D.body.appendChild(card);
  }

  function targets(step) {
    var t = step.target ? step.target() : null;
    if (!t) return [];
    return (Array.isArray(t) ? t : [t]).filter(visible);
  }
  function unionRect(els) {
    var r = null;
    els.forEach(function (e) {
      var b = e.getBoundingClientRect();
      r = r ? { left: Math.min(r.left, b.left), top: Math.min(r.top, b.top), right: Math.max(r.right, b.right),
                bottom: Math.max(r.bottom, b.bottom) } : { left: b.left, top: b.top, right: b.right, bottom: b.bottom };
    });
    return r;
  }

  function track() {
    cancelAnimationFrame(raf);
    var step = STEPS[S.i];
    (function loop() {
      if (!S.running) return;
      var els = targets(step), pad = 10;
      if (els.length) {
        var r = unionRect(els);
        if (!spot || spot.className !== "gdt-spot") { if (spot) spot.remove(); spot = el("div", "gdt-spot"); D.body.appendChild(spot); }
        spot.style.left = (r.left - pad) + "px"; spot.style.top = (r.top - pad) + "px";
        spot.style.width = (r.right - r.left + 2 * pad) + "px"; spot.style.height = (r.bottom - r.top + 2 * pad) + "px";
        var below = r.bottom + pad + 16, roomBelow = W.innerHeight - below;
        if (roomBelow > card.offsetHeight + 8) { card.style.top = below + "px"; card.style.bottom = "auto"; }
        else if (r.top - pad - 16 > card.offsetHeight + 8) { card.style.top = (r.top - pad - 16 - card.offsetHeight) + "px"; card.style.bottom = "auto"; }
        else { card.style.top = "auto"; card.style.bottom = "18px"; }
      } else {
        if (!spot || spot.className !== "gdt-dim") { if (spot) spot.remove(); spot = el("div", "gdt-dim"); D.body.appendChild(spot); }
        card.style.top = "auto"; card.style.bottom = "max(18px, calc(50vh - 110px))";
      }
      var f = card.querySelector(".gdt-fill");
      var done = S.paused ? 1 - S.left / (step.dur * 1000) : 1 - Math.max(0, S.left - (Date.now() - S.started)) / (step.dur * 1000);
      f.style.width = Math.min(100, Math.max(0, done * 100)) + "%";
      raf = requestAnimationFrame(loop);
    })();
  }

  function scrollTo(step) {
    var els = targets(step);
    if (!els.length) return;
    var r = unionRect(els), h = r.bottom - r.top;
    var anchor = els[0];
    anchor.scrollIntoView({ behavior: "smooth", block: h > W.innerHeight * 0.62 ? "start" : "center" });
  }

  async function go(i) {
    if (i < 0) i = 0;
    if (i >= STEPS.length) { stop(); return; }
    var token = ++S.token;
    clearTimeout(timer);
    S.i = i;
    var step = STEPS[i];
    card.querySelector(".gdt-k").textContent = (i + 1) + " / " + STEPS.length;
    card.querySelector(".gdt-t").textContent = step.title;
    card.querySelector(".gdt-x").textContent = step.text;
    if (step.before) { try { await step.before(); } catch (e) { /* keep going: a missing control must not stop the tour */ } }
    if (token !== S.token || !S.running) return;
    await waitFor(function () { return !step.target || targets(step).length; }, 6000);
    if (token !== S.token || !S.running) return;
    scrollTo(step);
    S.left = step.dur * 1000; S.started = Date.now(); S.deadline = Date.now() + S.left + 60000;
    track();
    if (!S.paused) schedule(token);
  }

  // Move on when the step's time is up and, for steps with until(), once it holds (at most 60 s extra).
  function schedule(token) {
    var step = STEPS[S.i];
    timer = setTimeout(function tick() {
      if (token !== S.token || S.paused) return;
      if (step.until && !step.until() && Date.now() < S.deadline) { timer = setTimeout(tick, 500); return; }
      if (step.until) { timer = setTimeout(function () { if (token === S.token && !S.paused) go(S.i + 1); }, 4000); return; }
      go(S.i + 1);
    }, S.left);
  }

  function togglePause() {
    if (!S.running) return;
    var b = card.querySelector('[data-a="pause"]');
    if (!S.paused) {
      S.paused = true; clearTimeout(timer); S.left = Math.max(0, S.left - (Date.now() - S.started));
      b.innerHTML = "&#9654;"; card.querySelector(".gdt-k").textContent = "Paused · " + (S.i + 1) + " / " + STEPS.length;
    } else {
      S.paused = false; S.started = Date.now(); b.innerHTML = "&#10073;&#10073;";
      card.querySelector(".gdt-k").textContent = (S.i + 1) + " / " + STEPS.length;
      S.deadline = Date.now() + S.left + 60000;
      schedule(S.token);
    }
  }

  function onKey(e) {
    if (!S.running) return;
    var tag = (e.target && e.target.tagName) || "";
    if (/INPUT|TEXTAREA|SELECT/.test(tag)) return;
    if (e.key === " " || e.key === "Spacebar") { e.preventDefault(); togglePause(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); go(S.i + 1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); go(S.i - 1); }
    else if (e.key === "Escape") { e.preventDefault(); stop(); }
  }

  function start() {
    if (S.running) return;
    S.running = true; S.paused = false; S.token++;
    launch.style.display = "none";
    buildCard();
    D.addEventListener("keydown", onKey, true);
    go(0);
  }

  function stop() {
    S.running = false; S.token++;
    clearTimeout(timer); cancelAnimationFrame(raf);
    if (spot) { spot.remove(); spot = null; }
    if (card) { card.remove(); card = null; }
    D.removeEventListener("keydown", onKey, true);
    launch.style.display = "";
  }

  launch.addEventListener("click", start);
  W.__gdtTour = { start: start, stop: stop, steps: STEPS };
})();

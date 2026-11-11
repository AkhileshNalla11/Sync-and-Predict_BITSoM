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
    { title: "Glycaemic Digital Twin · guided tour", dur: 10, target: null,
      text: "A 2-hour early warning for highs and lows in Type 2 diabetes. The tour plays by itself. Space pauses, the arrow keys step, Esc exits." },
    { title: "Pick a patient and a risk model", dur: 22,
      before: async function () { openSidebar(); await sleep(300); },
      target: function () { return D.querySelector('[data-testid="stSidebar"]'); },
      text: "Every virtual patient here is from the held-out test set: none was used for training, model selection or alert thresholds. The prediction time starts at this patient's highest-risk moment." },
    { title: "The patient, at the prediction time", dur: 20,
      before: async function () { await openTab("Live replay"); },
      target: function () { return all(".gt-hero")[0]; },
      text: "Profile from the synthetic health record: age, medicines, HbA1c, kidney function. The model sees only the data before this moment." },
    { title: "Two risks, refreshed every 30 minutes", dur: 30,
      target: function () { return all(".gt-risk"); },
      text: "The chance of a low (below 70 mg/dL) and a high (above 180 mg/dL) in the next 2 hours. The marker is the alert threshold, set on validation patients for a fixed false-alert rate: 1% for lows." },
    { title: "Live replay", dur: 12,
      before: async function () { await openTab("Live replay"); await waitFor(function () { return chart(); }); },
      target: function () { return box(chart()); },
      text: "The patient's record plays forward as if it were happening now. At every moment the model has seen only the data up to the moving line." },
    { title: "Watch for the alert", dur: 30,
      before: async function () {
        var b = await waitFor(function () { var c = chart(); return c && c.querySelector(".updatemenu-button"); }, 8000);
        click(b);
      },
      until: function () {                      // stay until the replay has played to its last frame
        var c = chart(), s = c && c._fullLayout && c._fullLayout.sliders && c._fullLayout.sliders[0];
        return !s || s.active >= s.steps.length - 1;
      },
      target: function () { return box(chart()); },
      text: "Risk climbs and the alert fires while glucose still looks normal. The status card names the top reasons. Then the low arrives, and the card shows how many minutes of warning there were." },
    { title: "Every alert is checked against what happened", dur: 20,
      target: function () { var s = section("What happens in this replay"); return s ? next(s, 1) : null; },
      text: "For each alert: how far ahead it warned, or whether it was a false alarm. Lows with no alert are listed as missed. Nothing is hidden." },
    { title: "Glucose timeline", dur: 35,
      before: async function () { await openTab("Glucose timeline"); await waitFor(function () { return chart(); }); },
      target: function () { return box(chart()); },
      text: "Shaded bands mark highs and lows. Filled triangles are alerts; open ones are repeats the policy held back, so there is no alarm flood. The dotted trace after NOW is what really happened; the model never saw it." },
    { title: "Why the alert fired", dur: 35,
      before: async function () { await openTab("Alert drivers"); await waitFor(function () { return chart(); }); },
      target: function () { return box(chart()); },
      text: "SHAP splits the risk into each input's contribution. It explains the model's reasoning, not the body's physiology." },
    { title: "In plain words, and by data stream", dur: 25,
      target: function () { var c = box(chart()); return c ? next(c, 2) : null; },
      text: "The top reasons as sentences a clinician can read, and how much the glucose sensor, the wearable and the health record each added." },
    { title: "Ask the twin “what if?”", dur: 20,
      before: async function () {
        await openTab("What-if");
        var start = await waitFor(function () { return byText("button", /^Start the what-if twin/, panel()); }, 3000);
        if (start) { start.click(); }
        var opt = await waitFor(function () { return byText('label[data-baseweb="radio"]', /Extra insulin dose/, panel()); }, 30000);
        if (opt) { opt.click(); }
        await waitFor(function () { return all(".gt-compare", panel()); }, 30000);
      },
      target: function () { var l = byText('label[data-baseweb="radio"]', /Extra insulin dose/, panel()); return l ? next(l.closest('[data-testid="stRadio"]') || l, 1) : null; },
      text: "A physiology model fitted to this patient's own first week of data. Here we try an extra insulin dose taken an hour ago." },
    { title: "Risk before and after the change", dur: 25,
      before: async function () { await openTab("What-if"); },
      target: function () { return all(".gt-compare", panel()); },
      text: "What the risk model predicts from the data it would see, before and after the change, for a low and for a high." },
    { title: "The next 2 hours, simulated", dur: 40,
      target: function () { var c = chart(); return c ? next(c, 1) : null; },
      text: "Dotted: the twin's estimate as things were logged. Solid: with the extra dose. The shaded 80% band is how far the twin usually misses; on test patients real glucose fell inside it 79% of the time." },
    { title: "Fitted to this patient", dur: 20,
      target: function () { var t = byText(".gt-chiprow", /Fit error/, panel()); return t ? next(t, 1) : null; },
      text: "Four parameters fitted to this patient's first week of glucose, meals and doses, with the fit error shown. No hidden simulator information is used." },
    { title: "Patient overview", dur: 20,
      before: async function () { await openTab("Patient overview"); await waitFor(function () { return section("Profile"); }); },
      target: function () { var s = section("Profile"); return s ? next(s, 3) : null; },
      text: "The health record the twin is built on: medicines, other conditions and the latest labs." },
    { title: "14 days of glucose against international targets", dur: 30,
      target: function () { var s = section("CGM summary"); return s ? next(s, 2) : null; },
      text: "Time in range, time below 70, time above 180, variability and GMI, each shown next to its consensus target." },
    { title: "How many alerts?", dur: 20,
      target: function () { var s = section("Alerts over the record"); return s ? next(s, 1) : null; },
      text: "Alerts per day for this patient over the whole record: one per risk episode, repeats suppressed." },
    { title: "A second model, trained on real patients", dur: 25,
      before: async function () { openSidebar(); await sleep(300); },
      target: function () { return sidebarWidget("Risk model"); },
      text: "Switch here to the model trained on 100 real patients (ShanghaiT2DM). On real data it warned of 35% of lows at least 30 minutes ahead, at about 0.6 alerts a day." },
    { title: "That's the twin", dur: 12, target: null,
      text: "Warns early. Explains itself. Honest about what it knows. Research prototype on synthetic and public data, not for clinical use." },
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
    ".gdt-card{position:fixed;left:50%;transform:translateX(-50%);z-index:999992;width:min(760px,calc(100vw - 32px));",
    "background:" + C.ink + ";color:#fff;border-radius:18px;padding:18px 20px 14px;font-family:" + FONT + ";",
    "box-shadow:0 18px 50px -12px rgba(0,0,0,.55);transition:top .35s ease,bottom .35s ease}",
    ".gdt-card .gdt-k{font-size:12px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:" + C.brandSoft + "}",
    ".gdt-card .gdt-t{font-size:22px;font-weight:700;margin:4px 0 6px;letter-spacing:-.01em}",
    ".gdt-card .gdt-x{font-size:16px;line-height:1.5;color:#dbe6ee}",
    ".gdt-card .gdt-bar{height:4px;border-radius:4px;background:rgba(255,255,255,.14);margin:14px 0 10px;overflow:hidden}",
    ".gdt-card .gdt-fill{height:100%;width:0;background:" + C.brandSoft + "}",
    ".gdt-card .gdt-row{display:flex;align-items:center;justify-content:space-between;gap:10px}",
    ".gdt-card .gdt-dots{display:flex;gap:5px;flex-wrap:wrap}",
    ".gdt-card .gdt-dot{width:7px;height:7px;border-radius:50%;background:rgba(255,255,255,.22)}",
    ".gdt-card .gdt-dot.on{background:" + C.brandSoft + "}",
    ".gdt-card button{background:rgba(255,255,255,.1);color:#fff;border:1px solid rgba(255,255,255,.18);border-radius:10px;",
    "padding:6px 11px;font:600 13px/1 " + FONT + ";cursor:pointer;margin-left:6px}",
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
      '<div class="gdt-k"></div><div class="gdt-t"></div><div class="gdt-x"></div>' +
      '<div class="gdt-bar"><div class="gdt-fill"></div></div>' +
      '<div class="gdt-row"><div class="gdt-dots"></div><div>' +
      '<button data-a="prev" title="Previous (←)">&#9664;</button>' +
      '<button data-a="pause" title="Pause / resume (Space)">&#10073;&#10073;</button>' +
      '<button data-a="next" title="Next (→)">&#9654;</button>' +
      '<button data-a="stop" title="Exit (Esc)">&#10005;</button></div></div>';
    var dots = card.querySelector(".gdt-dots");
    STEPS.forEach(function () { dots.appendChild(el("span", "gdt-dot")); });
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
    card.querySelector(".gdt-k").textContent = "Step " + (i + 1) + " of " + STEPS.length;
    card.querySelector(".gdt-t").textContent = step.title;
    card.querySelector(".gdt-x").textContent = step.text;
    Array.prototype.forEach.call(card.querySelectorAll(".gdt-dot"), function (d, k) { d.classList.toggle("on", k <= i); });
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
      b.innerHTML = "&#9654;"; card.querySelector(".gdt-k").textContent = "Paused · Step " + (S.i + 1) + " of " + STEPS.length;
    } else {
      S.paused = false; S.started = Date.now(); b.innerHTML = "&#10073;&#10073;";
      card.querySelector(".gdt-k").textContent = "Step " + (S.i + 1) + " of " + STEPS.length;
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

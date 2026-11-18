/* Stage demo for the clinician dashboard.
 *
 * Runs in the Streamlit page itself (injected once by src/app/tour.py). One click on "Start demo" plays the
 * finals demo script by itself, about 3 min 15 s: the patient, the live replay (pausing at the alert and at
 * the low), why the alert fired, the what-if twin, the 14-day overview and the real-patient model. There is
 * no caption bar: a soft spotlight shows the audience where to look while the presenter talks.
 *
 * Hidden controls, for emergencies or a presentation clicker: Space pauses, Right / Page Down skips ahead,
 * Left / Page Up goes back, Esc stops. It only clicks the dashboard's own controls.
 */
(function () {
  "use strict";
  var W = window, D = document;
  if (W.__gdtTour) return;

  var C = { brand: "#0f766e", ring: "#5eead4" };
  var FONT = "Geist, Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
  var DIM = "rgba(10,22,36,.55)";

  // ------------------------------------------------------------------ helpers
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  function visible(el) { if (!el) return false; var r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; }
  function all(sel, root) { return Array.prototype.slice.call((root || D).querySelectorAll(sel)).filter(visible); }
  function anyAll(sel, root) { return Array.prototype.slice.call((root || D).querySelectorAll(sel)); }
  function byText(sel, re, root) { return all(sel, root).filter(function (el) { return re.test(el.textContent.trim()); })[0] || null; }
  function anyText(sel, re, root) { return anyAll(sel, root).filter(function (el) { return re.test(el.textContent.trim()); })[0] || null; }
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
  function sidebar() { return D.querySelector('[data-testid="stSidebar"]'); }
  function sidebarWidget(label) {
    var sb = sidebar(), l = sb && byText('[data-testid="stWidgetLabel"]', new RegExp("^" + label), sb);
    return l ? box(l) : null;
  }
  async function waitFor(fn, ms) {
    var t = Date.now();
    while (Date.now() - t < (ms || 15000)) { var v = fn(); if (v && (!Array.isArray(v) || v.length)) return v; await sleep(150); }
    return null;
  }
  function running() {                          // Streamlit is re-running the script
    var s = D.querySelector('[data-testid="stStatusWidget"]');
    return !!(s && visible(s) && /running/i.test(s.textContent));
  }
  // Wait for a rerun to start (a click takes a moment to reach the server), then for the app to stay quiet.
  async function idle(ms) {
    var t = Date.now(), limit = ms || 30000, since = Date.now();
    await waitFor(running, 2500);
    while (Date.now() - t < limit) {           // wall-clock based: timers run slow in a background tab
      if (running()) since = Date.now();
      else if (Date.now() - since >= 1500) return;
      await sleep(150);
    }
  }
  async function openTab(name) {
    var b = await waitFor(function () { return tabButton(name); }, 20000);
    if (b && b.getAttribute("aria-selected") !== "true") { b.click(); await sleep(500); }
  }
  function click(el) { if (el) el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, view: W })); }
  function sidebarOpen() { var sb = sidebar(); return !!(sb && sb.getAttribute("aria-expanded") !== "false"); }
  function closeSidebar() {
    if (!sidebarOpen()) return;
    var b = D.querySelector('[data-testid="stSidebarCollapseButton"] button') ||
            D.querySelector('[data-testid="stSidebar"] button[kind="headerNoPadding"]');
    if (b) b.click();
  }
  function openSidebar() {
    if (sidebarOpen()) return;
    var b = D.querySelector('[data-testid="stSidebarCollapsedControl"] button') ||
            D.querySelector('[data-testid="stExpandSidebarButton"]');
    if (b) b.click();
  }
  function scrollTop() { var m = D.querySelector('[data-testid="stMain"]') || D.querySelector("section.main"); if (m) m.scrollTo({ top: 0, behavior: "smooth" }); W.scrollTo({ top: 0, behavior: "smooth" }); }

  // The live replay is a Plotly animation; these read its slider and press its own Play / Pause buttons.
  function replay() { return chart(); }
  function frameNow() {
    var c = replay(), s = c && c._fullLayout && c._fullLayout.sliders && c._fullLayout.sliders[0];
    return s ? s.steps[s.active].args[0][0] : null;
  }
  function frameIndex(name) {
    var c = replay(), s = c && c._fullLayout && c._fullLayout.sliders && c._fullLayout.sliders[0];
    if (!s) return -1;
    for (var i = 0; i < s.steps.length; i++) if (s.steps[i].args[0][0] === name) return i;
    return -1;
  }
  function replayButton(i) { var c = replay(); return c ? c.querySelectorAll(".updatemenu-button")[i] : null; }
  function frameActive() {
    var c = replay(), s = c && c._fullLayout && c._fullLayout.sliders && c._fullLayout.sliders[0];
    return s ? s.active : -1;
  }
  // Play, then pause on the frame named e.g. "08:25". Self-healing: if a rerun redraws the chart and the
  // animation stops, it presses Play again (and re-reads the chart, which may be a new element).
  async function playUntil(name) {
    var token = S.token, t0 = Date.now();
    if (frameIndex(name) < 0) return;
    while (frameActive() < frameIndex(name) && Date.now() - t0 < 120000) {
      click(replayButton(0));                   // Play (again, after a Space pause or a stall)
      var seen = frameActive(), moved = Date.now();
      while (!S.paused && frameActive() < frameIndex(name)) {
        await sleep(40);
        if (token !== S.token || !S.running) { click(replayButton(1)); throw new Error("skipped"); }
        if (frameActive() !== seen) { seen = frameActive(); moved = Date.now(); }
        else if (Date.now() - moved > 3000) {    // stalled: clear Plotly's frame queue, then press Play again
          click(replayButton(1)); await sleep(250); break;
        }
      }
      if (S.paused) { click(replayButton(1)); while (S.paused) { await sleep(100); if (token !== S.token || !S.running) throw new Error("skipped"); } }
    }
    click(replayButton(1));
  }

  // A 4-hour replay window keeps the night out of the replay (the select slider moves one notch per key).
  async function setReplayHours(want) {
    var lab = anyText('[data-testid="stWidgetLabel"]', /^Start this many hours/);
    var th = lab && box(lab).querySelector('[role="slider"]');
    if (!th || th.getAttribute("aria-valuetext") === String(want)) return;
    var left = Number(th.getAttribute("aria-valuetext")) > want;
    var key = left ? { key: "ArrowLeft", code: "ArrowLeft", keyCode: 37 } : { key: "ArrowRight", code: "ArrowRight", keyCode: 39 };
    th.focus();
    th.dispatchEvent(new KeyboardEvent("keydown", Object.assign({ bubbles: true }, key)));
    th.dispatchEvent(new KeyboardEvent("keyup", Object.assign({ bubbles: true }, key)));
    th.blur();
    await idle(30000);
  }

  // Done during the opening beat, while the presenter introduces the patient: the reruns finish off screen.
  async function prepareStage() {
    await setReplayHours(4);
    await prepareWhatIf();
  }

  // The what-if twin is started early, in the background, so its result is ready by the time it is shown.
  async function prepareWhatIf() {
    var start = anyText("button", /^Start the what-if twin/);
    if (start) { start.click(); await idle(60000); }
    var opt = await waitFor(function () { return anyText('label[data-baseweb="radio"]', /Extra insulin dose/); }, 30000);
    var input = opt && opt.querySelector("input");
    if (opt && !(input && input.checked)) { opt.click(); await idle(60000); }
  }

  // ------------------------------------------------------------------ the demo script (finals, about 3:15)
  // dur: seconds held after `before` and `run` finish. target: element(s) to spotlight (null = soft dim only).
  var STEPS = [
    { name: "Meet the patient", dur: 18,
      before: async function () { closeSidebar(); await openTab("Live replay"); scrollTop(); },
      after: function () { W.__gdtWhatIf = W.__gdtPrep || prepareStage(); },
      target: function () { return all(".gt-hero").concat(all(".gt-risk")); } },
    { name: "Live replay", dur: 4,
      before: async function () {
        if (W.__gdtWhatIf) { try { await W.__gdtWhatIf; } catch (e) { } }
        await openTab("Live replay"); await idle(30000);
        await waitFor(function () { return replay() && replayButton(0); }, 20000);
      },
      target: function () { return box(replay()); },
      run: async function () {
        await hold(4000);                       // "This replays her morning..."
        await playUntil("08:25"); await hold(11000);   // the alert fires
        await playUntil("09:45"); await hold(9000);    // below 70: 75 minutes of warning
      } },
    { name: "Why it fired", dur: 16,
      before: async function () { await openTab("Alert drivers"); await waitFor(function () { return chart(); }); },
      target: function () { return box(chart()); } },
    { name: "Reasons and their biology", dur: 13,
      target: function () { var c = box(chart()); return c ? next(c, 2) : null; } },
    { name: "Sensor quality", dur: 9,
      before: async function () { scrollTop(); await sleep(500); },
      target: function () { return all(".gt-chiprow").slice(0, 1).map(box); } },
    { name: "What if: extra insulin", dur: 20,
      before: async function () {
        await openTab("What-if");
        if (!all(".gt-compare", panel()).length) { await prepareWhatIf(); }
        await waitFor(function () { return all(".gt-compare", panel()); }, 60000);
      },
      target: function () { return all(".gt-compare", panel()); } },
    { name: "The twin's next 2 hours", dur: 33,
      target: function () { var c = chart(); return c ? next(c, 1) : null; } },
    { name: "14-day overview", dur: 20,
      before: async function () { await openTab("Patient overview"); await waitFor(function () { return section("CGM summary"); }); },
      target: function () { var s = section("CGM summary"); return s ? next(s, 2) : null; } },
    { name: "Real-patient model", dur: 11, noScroll: true,
      before: async function () {             // the widget sits near the top: snap the sidebar there, no scrolling
        openSidebar(); await sleep(500);
        var c = D.querySelector('[data-testid="stSidebarContent"]'); if (c) c.scrollTop = 0;
      },
      target: function () { return sidebarWidget("Risk model"); },
      run: async function () {
        await hold(3500);
        var opt = byText('label[data-baseweb="radio"]', /Real/i, sidebar());
        if (opt) opt.click();
      } },
  ];

  // ------------------------------------------------------------------ overlay (no caption bar)
  var css = D.createElement("style");
  css.textContent = [
    ".gdt-launch{position:fixed;right:22px;bottom:22px;z-index:999990;display:flex;align-items:center;gap:8px;",
    "padding:11px 18px;border:none;border-radius:999px;background:" + C.brand + ";color:#fff;font:600 14px/1 " + FONT + ";",
    "box-shadow:0 8px 24px -8px rgba(15,118,110,.7),0 2px 6px rgba(15,27,42,.18);cursor:pointer;transition:opacity .3s}",
    ".gdt-launch:hover{filter:brightness(1.08)}",
    ".gdt-spot{position:fixed;z-index:999991;border-radius:18px;pointer-events:none;opacity:0;",
    "box-shadow:0 0 0 3px " + C.ring + ",0 0 0 9999px " + DIM + ";",
    "transition:left .8s cubic-bezier(.2,.8,.2,1),top .8s cubic-bezier(.2,.8,.2,1),width .8s cubic-bezier(.2,.8,.2,1),",
    "height .8s cubic-bezier(.2,.8,.2,1),opacity .6s ease,box-shadow .6s ease}",
    ".gdt-spot.on{opacity:1}",
    ".gdt-spot.soft{box-shadow:0 0 0 0 transparent,0 0 0 9999px rgba(10,22,36,.18)}",
    ".gdt-pause{position:fixed;right:22px;bottom:22px;z-index:999992;padding:6px 10px;border-radius:8px;",
    "background:rgba(15,27,42,.75);color:#fff;font:600 12px/1 " + FONT + ";pointer-events:none;opacity:0;transition:opacity .3s}",
    ".gdt-pause.on{opacity:1}",
    // On stage, hide Streamlit's own "Running..." badge (still detected by running(): it keeps its size).
    "body.gdt-stage [data-testid=\"stStatusWidget\"]{visibility:hidden}",
  ].join("");
  D.head.appendChild(css);

  var launch = D.createElement("button");
  launch.className = "gdt-launch";
  launch.innerHTML = "&#9654;&nbsp; Start demo";
  launch.title = "Play the 3-minute stage demo (Space pauses, arrows skip, Esc stops)";
  D.body.appendChild(launch);

  var spot = null, badge = null, raf = 0;
  var S = { i: -1, paused: false, running: false, token: 0 };

  function el(tag, cls) { var e = D.createElement(tag); if (cls) e.className = cls; return e; }

  function targets(step) {
    var t = step && step.target ? step.target() : null;
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

  // Keep the spotlight on its target as the page scrolls or re-renders.
  function track() {
    cancelAnimationFrame(raf);
    (function loop() {
      if (!S.running) return;
      var els = targets(STEPS[S.i]), pad = 12;
      if (els.length) {
        var r = unionRect(els), first = !spot.getAttribute("data-placed");
        if (first) spot.style.transition = "none";      // appear on the first target, not slide in from a corner
        spot.classList.remove("soft");
        spot.style.left = (r.left - pad) + "px"; spot.style.top = (r.top - pad) + "px";
        spot.style.width = (r.right - r.left + 2 * pad) + "px"; spot.style.height = (r.bottom - r.top + 2 * pad) + "px";
        if (first) { spot.getBoundingClientRect(); spot.style.transition = ""; spot.setAttribute("data-placed", "1"); }
      } else {
        spot.classList.add("soft");
      }
      raf = requestAnimationFrame(loop);
    })();
  }

  function scrollTo(step) {
    var els = targets(step);
    if (step.noScroll || !els.length) return;
    var r = unionRect(els), h = r.bottom - r.top;
    els[0].style.scrollMarginTop = "28px";       // keep the spotlight ring on screen above a tall target
    els[0].scrollIntoView({ behavior: "smooth", block: h > W.innerHeight * 0.7 ? "start" : "center" });
  }

  // A pausable wait: Space freezes the clock, a skip ends it early.
  async function hold(ms) {
    var token = S.token, left = ms, last = Date.now();
    while (left > 0) {
      await sleep(100);
      if (token !== S.token || !S.running) throw new Error("skipped");
      var now = Date.now();
      if (!S.paused) left -= now - last;
      last = now;
    }
  }

  async function go(i) {
    if (i < 0) i = 0;
    if (i >= STEPS.length) { finish(); return; }
    var token = ++S.token;
    S.i = i;
    var step = STEPS[i];
    try {
      if (step.before) { try { await step.before(); } catch (e) { /* a missing control must not stop the demo */ } }
      if (token !== S.token || !S.running) return;
      await waitFor(function () { return !step.target || targets(step).length; }, 8000);
      if (token !== S.token || !S.running) return;
      scrollTo(step);
      track();
      if (step.after) { try { step.after(); } catch (e) { } }
      if (step.run) { await step.run(); }
      await hold(step.dur * 1000);
    } catch (e) { return; }                     // skipped or stopped: the newer step has taken over
    if (token === S.token && S.running) go(i + 1);
  }

  function togglePause() {
    if (!S.running) return;
    S.paused = !S.paused;
    badge.classList.toggle("on", S.paused);
  }

  function onKey(e) {
    if (!S.running) return;
    var tag = (e.target && e.target.tagName) || "";
    if (/INPUT|TEXTAREA|SELECT/.test(tag)) return;
    var k = e.key;
    if (k === " " || k === "Spacebar") { e.preventDefault(); togglePause(); }
    else if (k === "ArrowRight" || k === "PageDown") { e.preventDefault(); S.paused = false; badge.classList.remove("on"); go(S.i + 1); }
    else if (k === "ArrowLeft" || k === "PageUp") { e.preventDefault(); S.paused = false; badge.classList.remove("on"); go(S.i - 1); }
    else if (k === "Escape") { e.preventDefault(); stop(); }
  }

  function start() {
    if (S.running) return;
    S.running = true; S.paused = false; S.token++;
    launch.style.display = "none";
    D.body.classList.add("gdt-stage");
    spot = el("div", "gdt-spot soft"); D.body.appendChild(spot);
    badge = el("div", "gdt-pause"); badge.textContent = "Paused"; D.body.appendChild(badge);
    requestAnimationFrame(function () { spot.classList.add("on"); });
    D.addEventListener("keydown", onKey, true);
    go(0);
  }

  function teardown() {
    S.running = false; S.token++;
    cancelAnimationFrame(raf);
    if (badge) { badge.remove(); badge = null; }
    D.body.classList.remove("gdt-stage");
    D.removeEventListener("keydown", onKey, true);
    launch.style.display = "";
  }

  function finish() {                           // fade the spotlight out and leave the last view on screen
    var s = spot; spot = null;
    teardown();
    if (s) { s.classList.remove("on"); setTimeout(function () { s.remove(); }, 700); }
  }

  function stop() {
    if (spot) { spot.remove(); spot = null; }
    teardown();
  }

  launch.addEventListener("click", start);
  W.__gdtTour = { start: start, stop: stop, steps: STEPS, state: S };

  // Stage mode: open the dashboard with ?stage=1 before the talk. The page then prepares itself (4-hour replay
  // window, what-if twin with the extra insulin dose) while the slides are on, so Start demo begins at once.
  // The button reads "Preparing demo…" until it is ready.
  if (/[?&]stage(=1|=true|&|$)/i.test(W.location.search)) {
    launch.innerHTML = "Preparing demo&hellip;";
    launch.style.opacity = ".65";
    W.__gdtPrep = (async function () {
      await waitFor(function () { return all(".gt-hero").length && replay(); }, 120000);
      await idle(60000);
      try { await prepareStage(); } catch (e) { }
      await idle(60000);
      launch.innerHTML = "&#9654;&nbsp; Start demo";
      launch.style.opacity = "";
    })();
  }
})();

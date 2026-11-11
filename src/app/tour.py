"""Guided tour: injects src/app/tour.js into the dashboard page once.

The script adds a "Guided tour" button (bottom right). Pressed, it walks through the dashboard by itself:
switches tabs, plays the replay, starts the what-if twin, spotlights one feature at a time and captions it.
Space pauses, the arrow keys step, Esc exits. It runs entirely in the browser, so it costs no server reruns
except the ones the what-if twin itself needs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import streamlit.components.v1 as components

TOUR_FILE = Path(__file__).with_name("tour.js")

# Runs inside the component's (same-origin) iframe and copies the tour into the parent page, so the tour keeps
# running while Streamlit re-renders the app. Re-installs only when tour.js changes.
_LOADER = """<script>
(function () {
  var W = window.parent, D = W.document;
  if (W.__gdtTourVersion === %(version)s) return;
  if (W.__gdtTour) { try { W.__gdtTour.stop(); } catch (e) {} }
  D.querySelectorAll('.gdt-launch').forEach(function (b) { b.remove(); });
  W.__gdtTour = undefined;
  W.__gdtTourVersion = %(version)s;
  var s = D.createElement('script');
  s.textContent = %(source)s;
  D.head.appendChild(s);
})();
</script>"""


def render() -> None:
    """Install the guided tour in the page (idempotent; renders a zero-height component).

    tour.js is read on every run, not at import: Streamlit only reloads changed .py modules, so a cached copy
    would keep serving an old tour after tour.js alone is updated on the server.
    """
    source = TOUR_FILE.read_text(encoding="utf-8")
    version = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
    components.html(_LOADER % {"version": json.dumps(version), "source": json.dumps(source)}, height=0)

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

TOUR_JS = (Path(__file__).with_name("tour.js")).read_text(encoding="utf-8")
VERSION = hashlib.sha1(TOUR_JS.encode("utf-8")).hexdigest()[:10]

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
    """Install the guided tour in the page (idempotent; renders a zero-height component)."""
    components.html(_LOADER % {"version": json.dumps(VERSION), "source": json.dumps(TOUR_JS)}, height=0)

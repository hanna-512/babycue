"""
BabyCue — AI-Powered Infant Cry Analysis Platform
Tagline: "Every little cry has a cue."

Backend Pipeline:
Pre-Trained YAMNet Embeddings (2,048-D) + PCA(64) + Class-Balanced Logistic Regression
Audited Unseen 5-Fold Cross-Validation Accuracy: 67.55% (638 Multi-Format Clips) | 69.45% (586 High-SNR WAV Clips)
"""

import base64
import csv
from datetime import date, datetime
import io
import os
import random
import subprocess
import tempfile
import time
import urllib.parse
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio_ffmpeg
import joblib
import librosa
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

if hasattr(tf, "__internal__") and not hasattr(tf.__internal__, "register_load_context_function"):
    tf.__internal__.register_load_context_function = getattr(
        tf.__internal__, "register_call_context_function", lambda func: None
    )

import tensorflow_hub as hub

# ---------------------------------------------------------------------------
# Paths & Core AI Configuration (Preserved)
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "best_model.pkl"
TELEMETRY_PATH = BASE / "logs" / "tester_feedback.csv"
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
TARGET_SR = 16000
MAX_FILE_MB = 10
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

AGE_CATEGORIES = [
    "Newborn: 0–3 months",
    "Young Infant: 3–6 months",
    "Older Infant: 6–12 months",
    "Young Toddler: 12–18 months",
]

LABELS = {
    "Hunger": "Hunger",
    "Pain_Discomfort": "Pain / Discomfort",
    "Sleepiness": "Sleepiness",
}
CLASS_COLORS = {
    "Hunger": "#C994A2",
    "Pain_Discomfort": "#765A63",
    "Sleepiness": "#DFB0BC",
}


def render_html(raw_html: str):
    """Strips leading indentation so Streamlit Markdown never renders HTML as a code block."""
    cleaned = "\n".join(line.strip() for line in raw_html.strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def compute_age_from_birthday(bday: date) -> tuple[str, str]:
    """Calculates human-readable infant age and maps it to one of the 4 developmental categories."""
    today = date.today()
    days = max(0, (today - bday).days)
    months = days / 30.4375

    if months < 1.0:
        weeks = max(1, days // 7)
        readable = f"{weeks} week{'s' if weeks != 1 else ''} old"
    elif months < 12.0:
        m_int = int(months)
        rem_weeks = int((days - m_int * 30.4375) // 7)
        if rem_weeks > 0:
            readable = f"{m_int} mo{'s' if m_int != 1 else ''}, {rem_weeks} wk{'s' if rem_weeks != 1 else ''}"
        else:
            readable = f"{m_int} month{'s' if m_int != 1 else ''} old"
    else:
        m_int = int(round(months))
        readable = f"{m_int} months old"

    if months < 3.0:
        cat = AGE_CATEGORIES[0]
    elif months < 6.0:
        cat = AGE_CATEGORIES[1]
    elif months < 12.0:
        cat = AGE_CATEGORIES[2]
    else:
        cat = AGE_CATEGORIES[3]

    return readable, cat


# ---------------------------------------------------------------------------
# Lucide SVG Icon Library (Styled in Muted Mauve #765A63 & Dusty Rose #C994A2)
# ---------------------------------------------------------------------------
LUCIDE_PATHS = {
    "house": '<path d="M15 21v-8a1 1 0 0 0-1-1h-4a1 1 0 0 0-1 1v8"/><path d="M3 10a2 2 0 0 1 .709-1.528l7-5.999a2 2 0 0 1 2.582 0l7 5.999A2 2 0 0 1 21 10v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "audio-lines": '<path d="M2 10v3"/><path d="M6 6v11"/><path d="M10 3v18"/><path d="M14 8v7"/><path d="M18 5v13"/><path d="M22 10v3"/>',
    "chart": '<path d="M12 16v5"/><path d="M16 14v7"/><path d="M20 10v11"/><path d="m22 3-8.646 8.646a.5.5 0 0 1-.708 0L9.354 8.354a.5.5 0 0 0-.707 0L2 15"/><path d="M4 18v3"/><path d="M8 14v7"/>',
    "history": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>',
    "book-open": '<path d="M12 7v14"/><path d="M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z"/>',
    "messages-square": '<path d="M14 9a2 2 0 0 1-2 2H6l-4 4V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2z"/><path d="M18 9h2a2 2 0 0 1 2 2v11l-4-4h-6a2 2 0 0 1-2-2v-1"/>',
    "settings": '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>',
    "user-round": '<circle cx="12" cy="8" r="5"/><path d="M20 21a8 8 0 0 0-16 0"/>',
    "mic": '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>',
    "heart": '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>',
    "moon": '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
    "sparkles": '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/>',
    "shield-check": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "brain": '<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/>',
    "lightbulb": '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>',
    "check-circle": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/>',
    "volume-2": '<path d="M11 4.702a.705.705 0 0 0-1.203-.498L6.413 7.587A1.4 1.4 0 0 1 5.416 8H3a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2.416a1.4 1.4 0 0 1 .997.413l3.383 3.384A.705.705 0 0 0 11 19.298z"/><path d="M16 9a5 5 0 0 1 0 6"/><path d="M19.364 18.364a9 9 0 0 0 0-12.728"/>',
    "baby-bottle": '<path d="M10 2h4"/><path d="M11 2v2.5L9 6.5V20a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2V6.5l-2-2V2"/><path d="M9 11h6"/><path d="M9 15h4"/>',
}


def lucide_svg(name: str, size: int = 18, color: str = "#765A63", stroke_width: float = 1.85) -> str:
    paths = LUCIDE_PATHS.get(name, LUCIDE_PATHS["sparkles"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" '
        f'stroke-linejoin="round" style="display:inline-block;vertical-align:middle;flex-shrink:0;">'
        f"{paths}</svg>"
    )


def lucide_b64_uri(name: str, color: str = "#765A63") -> str:
    """Base64-encoded SVG URI guaranteed to render in CSS ::before across all browsers."""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="1.95" stroke-linecap="round" stroke-linejoin="round">'
        f"{LUCIDE_PATHS[name]}</svg>"
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def babycue_logo_svg(size: int = 62) -> str:
    """Standalone BabyCue brand mark: soft blush circular halo + delicate acoustic soundwave."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 64 64" fill="none">'
        '<circle cx="32" cy="32" r="30" fill="#F4E2E6"/>'
        '<circle cx="32" cy="32" r="24" stroke="#E8C5CD" stroke-width="1.2" stroke-dasharray="3 3"/>'
        '<line x1="16" y1="29" x2="16" y2="35" stroke="#C994A2" stroke-width="2.4" stroke-linecap="round"/>'
        '<line x1="21" y1="24" x2="21" y2="40" stroke="#765A63" stroke-width="2.4" stroke-linecap="round"/>'
        '<line x1="26.5" y1="19" x2="26.5" y2="45" stroke="#C994A2" stroke-width="2.6" stroke-linecap="round"/>'
        '<line x1="32" y1="15" x2="32" y2="49" stroke="#765A63" stroke-width="2.8" stroke-linecap="round"/>'
        '<line x1="37.5" y1="20" x2="37.5" y2="44" stroke="#C994A2" stroke-width="2.6" stroke-linecap="round"/>'
        '<line x1="43" y1="25" x2="43" y2="39" stroke="#765A63" stroke-width="2.4" stroke-linecap="round"/>'
        '<line x1="48" y1="29" x2="48" y2="35" stroke="#C994A2" stroke-width="2.4" stroke-linecap="round"/>'
        "</svg>"
    )


def babycue_favicon_uri() -> str:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none">'
        '<circle cx="32" cy="32" r="30" fill="#F4E2E6"/>'
        '<line x1="20" y1="25" x2="20" y2="39" stroke="#765A63" stroke-width="3.2" stroke-linecap="round"/>'
        '<line x1="28" y1="18" x2="28" y2="46" stroke="#C994A2" stroke-width="3.4" stroke-linecap="round"/>'
        '<line x1="36" y1="16" x2="36" y2="48" stroke="#765A63" stroke-width="3.4" stroke-linecap="round"/>'
        '<line x1="44" y1="24" x2="44" y2="40" stroke="#C994A2" stroke-width="3.2" stroke-linecap="round"/>'
        "</svg>"
    )
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg)


def decorative_hero_art_svg() -> str:
    """Minimal line-art baby accents (teddy silhouette, small bottle, stars, heart) in muted mauve & blush."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="240" height="125" viewBox="0 0 240 125" fill="none" style="opacity:0.88;">'
        '<path d="M12 96 C 65 112, 108 62, 145 72 C 175 80, 192 52, 226 42" stroke="#DFC3CA" stroke-width="1.4" stroke-dasharray="4 4"/>'
        '<g transform="translate(52, 14)">'
        '<circle cx="11" cy="11" r="6.5" stroke="#C994A2" stroke-width="1.5" fill="#FDF9FA"/>'
        '<circle cx="35" cy="11" r="6.5" stroke="#C994A2" stroke-width="1.5" fill="#FDF9FA"/>'
        '<circle cx="23" cy="23" r="14" stroke="#C994A2" stroke-width="1.6" fill="#FFFFFF"/>'
        '<ellipse cx="23" cy="26.5" rx="5.5" ry="4.2" fill="#F4E2E6"/>'
        '<circle cx="18.5" cy="20.5" r="1.3" fill="#765A63"/>'
        '<circle cx="27.5" cy="20.5" r="1.3" fill="#765A63"/>'
        '<path d="M21.5 25.2 Q23 26.8 24.5 25.2" stroke="#765A63" stroke-width="1.3" stroke-linecap="round"/>'
        "</g>"
        '<g transform="translate(166, 36) rotate(22)">'
        '<path d="M11 0 C11 -3, 17 -3, 17 0 L18 5 L10 5 Z" stroke="#C994A2" stroke-width="1.4" fill="#F4E2E6"/>'
        '<rect x="8" y="5" width="12" height="4" rx="2" stroke="#C994A2" stroke-width="1.4" fill="#FFFFFF"/>'
        '<rect x="6" y="9" width="16" height="26" rx="5" stroke="#C994A2" stroke-width="1.5" fill="#FFFFFF"/>'
        '<line x1="10" y1="16" x2="14" y2="16" stroke="#DFC3CA" stroke-width="1.3" stroke-linecap="round"/>'
        '<line x1="10" y1="21" x2="15" y2="21" stroke="#DFC3CA" stroke-width="1.3" stroke-linecap="round"/>'
        '<line x1="10" y1="26" x2="13" y2="26" stroke="#DFC3CA" stroke-width="1.3" stroke-linecap="round"/>'
        "</g>"
        '<path d="M126 24 C126 21.5, 122.5 20.5, 121 23 C119.5 20.5, 116 21.5, 116 24 C116 27.5, 121 30.5, 121 30.5 C121 30.5, 126 27.5, 126 24 Z" fill="#E8C5CD"/>'
        '<path d="M150 18 L151.6 22.4 L156 24 L151.6 25.6 L150 30 L148.4 25.6 L144 24 L148.4 22.4 Z" fill="#DFC3CA"/>'
        '<path d="M214 78 L215.2 81.2 L218.5 82.5 L215.2 83.8 L214 87 L212.8 83.8 L209.5 82.5 L212.8 81.2 Z" fill="#E8C5CD"/>'
        "</svg>"
    )


def bottle_corner_svg() -> str:
    """Delicate milk bottle line-art accent for the care card."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="52" height="58" viewBox="0 0 44 52" fill="none">'
        '<g transform="translate(14, 4) rotate(24)">'
        '<path d="M9 0 C9 -2.5, 15 -2.5, 15 0 L16 5 L8 5 Z" stroke="#B67D8C" stroke-width="1.4" fill="#FFFFFF"/>'
        '<rect x="6" y="5" width="12" height="3.8" rx="1.9" stroke="#B67D8C" stroke-width="1.4" fill="#F4E2E6"/>'
        '<rect x="4" y="8.8" width="16" height="25" rx="5" stroke="#B67D8C" stroke-width="1.5" fill="#FFFFFF"/>'
        '<line x1="8" y1="15.5" x2="12" y2="15.5" stroke="#C994A2" stroke-width="1.3" stroke-linecap="round"/>'
        '<line x1="8" y1="20.5" x2="13.5" y2="20.5" stroke="#C994A2" stroke-width="1.3" stroke-linecap="round"/>'
        '<line x1="8" y1="25.5" x2="11.5" y2="25.5" stroke="#C994A2" stroke-width="1.3" stroke-linecap="round"/>'
        "</g>"
        "</svg>"
    )


def waveform_bars_svg(samples=None, height: int = 46, bar_count: int = 44) -> str:
    """Renders an elegant minimal soundwave bar graphic in Dusty Rose & Soft Blush."""
    if not samples or len(samples) < 5:
        pattern = [
            0.22, 0.35, 0.55, 0.42, 0.72, 0.88, 0.50, 0.32, 0.64, 0.92, 0.78,
            0.40, 0.28, 0.60, 0.85, 0.96, 0.68, 0.44, 0.76, 0.84, 0.52, 0.30,
            0.58, 0.90, 0.74, 0.46, 0.66, 0.82, 0.48, 0.26, 0.54, 0.70, 0.38,
            0.24, 0.48, 0.62, 0.36, 0.22, 0.40, 0.52, 0.30, 0.20, 0.32, 0.18,
        ]
        vals = pattern[:bar_count]
    else:
        arr = np.abs(np.array(samples, dtype=float))
        chunks = np.array_split(arr, bar_count)
        vals = [float(np.mean(c)) if len(c) > 0 else 0.15 for c in chunks]
        max_v = max(vals) if max(vals) > 0 else 1.0
        vals = [max(0.14, min(1.0, v / max_v)) for v in vals]

    width = bar_count * 7
    bars = []
    for i, v in enumerate(vals):
        bar_h = max(6, int(v * (height - 6)))
        y = (height - bar_h) // 2
        x = i * 7 + 2
        fill = "#C994A2" if v > 0.48 else "#E5C1CA"
        bars.append(f'<rect x="{x}" y="{y}" width="3.2" height="{bar_h}" rx="1.6" fill="{fill}"/>')

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="{height}" '
        f'viewBox="0 0 {width} {height}" preserveAspectRatio="none">{"".join(bars)}</svg>'
    )


# ---------------------------------------------------------------------------
# Page Config (Must be first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BabyCue — Every little cry has a cue.",
    page_icon=babycue_favicon_uri(),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session-Isolated State Initialization
# ---------------------------------------------------------------------------
NAV_ITEMS = [
    ("Home", "house"),
    ("Insights", "chart"),
    ("Care Guides", "book-open"),
    ("Parenting FAQ", "messages-square"),
    ("Settings", "settings"),
    ("Profile", "user-round"),
]
VALID_PAGES = [item[0] for item in NAV_ITEMS]

if "nav" not in st.session_state or st.session_state["nav"] not in VALID_PAGES:
    st.session_state["nav"] = "Home"
if "baby_name" not in st.session_state:
    st.session_state["baby_name"] = ""
if "baby_bday" not in st.session_state:
    st.session_state["baby_bday"] = None
if "baby_age" not in st.session_state or st.session_state["baby_age"] not in AGE_CATEGORIES:
    st.session_state["baby_age"] = AGE_CATEGORIES[0]
if "privacy_save" not in st.session_state:
    st.session_state["privacy_save"] = True
if "cry_history" not in st.session_state:
    st.session_state["cry_history"] = []
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "rated_ids" not in st.session_state:
    st.session_state["rated_ids"] = {}
if "routine_items" not in st.session_state:
    st.session_state["routine_items"] = [
        {"task": "Morning Feeding & Upright Burp", "time": "07:30 AM", "category": "Feeding", "done": True},
        {"task": "First Morning Nap Window", "time": "09:15 AM", "category": "Sleep", "done": False},
        {"task": "Diaper & Comfort Check", "time": "11:00 AM", "category": "Care", "done": False},
        {"task": "Midday Milk Feed", "time": "12:30 PM", "category": "Feeding", "done": False},
        {"task": "Warm Bath & Evening Wind-Down", "time": "07:00 PM", "category": "Sleep", "done": False},
    ]
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {
            "role": "assistant",
            "text": (
                "Welcome to BabyCue Support. Select any common parenting question below or ask about "
                "feeding cues, trapped gas relief, sleep windows, or how our acoustic analysis works."
            ),
        }
    ]


def navigate_to(target_page: str):
    st.session_state["nav"] = target_page


# ---------------------------------------------------------------------------
# Global Design System CSS (Typography, Exact Palette, Sidebar, Lucide Icons)
# ---------------------------------------------------------------------------
ICON_HOME_B64 = lucide_b64_uri("house", "#765A63")
ICON_INSIGHTS_B64 = lucide_b64_uri("chart", "#765A63")
ICON_GUIDES_B64 = lucide_b64_uri("book-open", "#765A63")
ICON_FAQ_B64 = lucide_b64_uri("messages-square", "#765A63")
ICON_SETTINGS_B64 = lucide_b64_uri("settings", "#765A63")
ICON_PROFILE_B64 = lucide_b64_uri("user-round", "#765A63")

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;1,9..144,400;1,9..144,500&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

:root {{
    --bg-ivory: #F8F3F1;
    --blush: #E8C5CD;
    --blush-soft: #F4E2E6;
    --dusty-rose: #C994A2;
    --dusty-rose-hover: #B67D8C;
    --mauve: #765A63;
    --charcoal: #343035;
    --muted-grey: #81777A;
    --white: #FFFFFF;
    --border-soft: #EAD9DD;
}}

html, body, .stApp {{
    background-color: var(--bg-ivory) !important;
    color: var(--charcoal) !important;
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

.block-container {{
    max-width: 1140px;
    padding-top: 1.4rem;
    padding-bottom: 3.5rem;
}}

/* Keep top header transparent BUT ensure the collapsed-sidebar reopen button (>>) stays visible! */
header[data-testid="stHeader"] {{
    background: transparent !important;
}}
[data-testid="stExpandSidebarButton"],
[data-testid="collapsedControl"] {{
    visibility: visible !important;
    opacity: 1 !important;
    background-color: var(--blush-soft) !important;
    border: 1px solid var(--blush) !important;
    border-radius: 12px !important;
    color: var(--mauve) !important;
}}

h1, h2, h3, h4 {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--charcoal) !important;
    letter-spacing: -0.02em;
}}

/* Sidebar Styling */
section[data-testid="stSidebar"] {{
    background-color: #FCF9F8 !important;
    border-right: 1px solid var(--border-soft) !important;
}}
section[data-testid="stSidebar"] .block-container {{
    padding-top: 1.3rem;
    padding-left: 1.05rem;
    padding-right: 1.05rem;
}}

/* Sidebar Navigation Buttons with Guaranteed Base64 Lucide SVG Icons */
section[data-testid="stSidebar"] div[data-testid="stButton"] button {{
    width: 100% !important;
    justify-content: flex-start !important;
    text-align: left !important;
    border-radius: 14px !important;
    padding: 0.62rem 0.95rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.93rem !important;
    box-shadow: none !important;
    transform: none !important;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"] {{
    background-color: transparent !important;
    border: 1px solid transparent !important;
    color: var(--charcoal) !important;
    font-weight: 600 !important;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"]:hover {{
    background-color: #F6ECEE !important;
    color: var(--mauve) !important;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {{
    background-color: var(--blush-soft) !important;
    border: 1px solid #E7C7CF !important;
    color: var(--charcoal) !important;
    font-weight: 700 !important;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] button p {{
    display: inline-flex !important;
    align-items: center !important;
    gap: 11px !important;
    margin: 0 !important;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] button p::before {{
    content: "";
    display: inline-block;
    width: 18px;
    height: 18px;
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    flex-shrink: 0;
}}
/* Map each sidebar navigation button (1..6) to its distinct Lucide SVG icon */
section[data-testid="stSidebar"] div.stElementContainer:has(button):nth-of-type(2) button p::before {{
    background-image: url("{ICON_HOME_B64}");
}}
section[data-testid="stSidebar"] div.stElementContainer:has(button):nth-of-type(3) button p::before {{
    background-image: url("{ICON_INSIGHTS_B64}");
}}
section[data-testid="stSidebar"] div.stElementContainer:has(button):nth-of-type(4) button p::before {{
    background-image: url("{ICON_GUIDES_B64}");
}}
section[data-testid="stSidebar"] div.stElementContainer:has(button):nth-of-type(5) button p::before {{
    background-image: url("{ICON_FAQ_B64}");
}}
section[data-testid="stSidebar"] div.stElementContainer:has(button):nth-of-type(6) button p::before {{
    background-image: url("{ICON_SETTINGS_B64}");
}}
section[data-testid="stSidebar"] div.stElementContainer:has(button):nth-of-type(7) button p::before {{
    background-image: url("{ICON_PROFILE_B64}");
}}

/* Main Canvas Primary & Secondary Buttons */
.main div.stButton > button[kind="primary"],
div[data-testid="stAppViewBlockContainer"] div.stButton > button[kind="primary"] {{
    background-color: var(--dusty-rose) !important;
    color: var(--white) !important;
    border: 1px solid var(--dusty-rose) !important;
    border-radius: 999px !important;
    padding: 0.62rem 1.55rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    box-shadow: 0 6px 18px rgba(201, 148, 162, 0.28) !important;
    transition: all 0.18s ease !important;
}}
.main div.stButton > button[kind="primary"]:hover,
div[data-testid="stAppViewBlockContainer"] div.stButton > button[kind="primary"]:hover {{
    background-color: var(--dusty-rose-hover) !important;
    border-color: var(--dusty-rose-hover) !important;
    transform: translateY(-1px);
}}
div[data-testid="stAppViewBlockContainer"] div.stButton > button[kind="secondary"],
div[data-testid="stPopover"] > button {{
    background-color: var(--white) !important;
    color: var(--charcoal) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 999px !important;
    padding: 0.52rem 1.25rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: all 0.16s ease !important;
}}
div[data-testid="stAppViewBlockContainer"] div.stButton > button[kind="secondary"]:hover,
div[data-testid="stPopover"] > button:hover {{
    border-color: var(--dusty-rose) !important;
    background-color: #FDF9FA !important;
    color: var(--mauve) !important;
}}

/* Download Button */
div.stDownloadButton > button {{
    background-color: var(--blush-soft) !important;
    color: var(--charcoal) !important;
    border: 1px solid var(--blush) !important;
    border-radius: 999px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
}}

/* File Uploader & Audio Recorder */
div[data-testid="stFileUploader"], div[data-testid="stAudioInput"] {{
    background-color: #FCF9F8 !important;
    border: 1.5px dashed #DEC4CB !important;
    border-radius: 18px !important;
    padding: 0.85rem 1rem !important;
}}

/* Tabs — Override Streamlit's Default Red Highlight Bar with Dusty Rose */
div[data-testid="stTabs"] button[data-baseweb="tab"] {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    color: var(--muted-grey) !important;
}}
div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--charcoal) !important;
}}
div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {{
    background-color: var(--dusty-rose) !important;
}}
div[data-testid="stTabs"] div[data-baseweb="tab-border"] {{
    background-color: var(--border-soft) !important;
}}

/* Progress Bars */
div[data-testid="stProgress"] > div > div > div > div {{
    background-color: var(--dusty-rose) !important;
}}

/* Reusable Surface Cards */
.bc-card {{
    background: var(--white);
    border: 1px solid var(--border-soft);
    border-radius: 22px;
    padding: 22px 24px;
    box-shadow: 0 8px 26px rgba(118, 90, 99, 0.045);
    margin-bottom: 16px;
}}
.bc-card-blush {{
    background: linear-gradient(145deg, #F6E5E9 0%, #EFE0E4 100%);
    border: 1px solid #E6C7CF;
    border-radius: 22px;
    padding: 22px 24px;
    box-shadow: 0 8px 24px rgba(118, 90, 99, 0.04);
    margin-bottom: 16px;
}}
.bc-pill-badge {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: var(--blush-soft);
    color: var(--mauve);
    border: 1px solid #E7C8D0;
    border-radius: 999px;
    padding: 5px 14px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}}
.bc-hero-title {{
    font-family: 'Fraunces', Georgia, serif;
    font-size: clamp(2.1rem, 4vw, 3.15rem);
    font-weight: 500;
    line-height: 1.12;
    color: var(--charcoal);
    margin: 14px 0 12px;
    letter-spacing: -0.02em;
}}
.bc-hero-accent {{
    color: var(--dusty-rose);
    font-style: italic;
}}
.bc-hero-sub {{
    font-family: 'DM Sans', sans-serif;
    font-size: 1.03rem;
    line-height: 1.6;
    color: var(--muted-grey);
    max-width: 540px;
}}
.bc-section-label {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--charcoal);
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
}}
.bc-step-circle {{
    width: 58px;
    height: 58px;
    border-radius: 50%;
    background: var(--blush-soft);
    border: 1px solid #E8C5CD;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 12px;
}}
.bc-kpi-card {{
    background: var(--white);
    border: 1px solid var(--border-soft);
    border-radius: 20px;
    padding: 18px 20px;
    box-shadow: 0 6px 20px rgba(118, 90, 99, 0.04);
}}
.bc-kpi-val {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--charcoal);
    margin: 6px 0 2px;
}}
.bc-kpi-lbl {{
    font-size: 0.83rem;
    color: var(--muted-grey);
    font-weight: 500;
}}
.bc-chat-bot {{
    background: var(--white);
    border: 1px solid var(--border-soft);
    border-radius: 20px 20px 20px 6px;
    padding: 14px 18px;
    margin-bottom: 10px;
    color: var(--charcoal);
    line-height: 1.55;
    box-shadow: 0 4px 14px rgba(118, 90, 99, 0.035);
}}
.bc-chat-user {{
    background: var(--dusty-rose);
    border-radius: 20px 20px 6px 20px;
    padding: 12px 18px;
    margin-bottom: 10px;
    color: var(--white);
    font-weight: 500;
    text-align: right;
}}
.bc-disclaimer {{
    background: #FCF9F8;
    border-left: 3px solid var(--dusty-rose);
    border-radius: 10px;
    padding: 11px 15px;
    font-size: 0.83rem;
    color: var(--muted-grey);
    line-height: 1.5;
    margin-top: 14px;
}}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Backend AI Pipeline & Utilities (Preserved 100%)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_ai_pipeline():
    if not MODEL_PATH.exists():
        return None, None
    classifier = joblib.load(MODEL_PATH)
    yamnet = hub.load(YAMNET_URL)
    return classifier, yamnet


model, yamnet_model = load_ai_pipeline()


def load_audio_standardized(audio_path, target_sr=TARGET_SR):
    """Decodes WAV natively or .3gp/.mp3/.ogg/.m4a via bundled FFmpeg at 16 kHz mono."""
    ext = Path(audio_path).suffix.lower()
    if ext == ".wav":
        y, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)
        return y

    tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        cmd = [
            FFMPEG_EXE,
            "-y",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            "-loglevel",
            "error",
            tmp_wav,
        ]
        subprocess.run(cmd, check=True)
        y, _ = librosa.load(tmp_wav, sr=target_sr, mono=True)
        return y
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


def extract_live_yamnet_features(audio_path, yamnet):
    """Matches Step 1 & Step 2 preprocessing and 2,048-D YAMNet embedding extraction."""
    y = load_audio_standardized(audio_path, target_sr=TARGET_SR)

    raw_duration = len(y) / TARGET_SR
    if raw_duration > 30.0:
        raise ValueError(f"Audio is too long ({raw_duration:.1f}s). Please keep recordings under 30 seconds.")

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    y_trimmed, _ = librosa.effects.trim(y, top_db=28)

    if len(y_trimmed) < int(0.8 * TARGET_SR):
        raise ValueError("Cry audio is too short or quiet (requires at least 0.8 seconds of clear cry sound).")

    y_float = y_trimmed.astype(np.float32)
    if len(y_float) < TARGET_SR:
        y_float = np.pad(y_float, (0, TARGET_SR - len(y_float)))

    _, embeddings, _ = yamnet(y_float)
    emb_np = embeddings.numpy()
    feature_vec = np.concatenate([np.mean(emb_np, axis=0), np.max(emb_np, axis=0)]).astype(np.float64)

    return feature_vec.reshape(1, -1), y_trimmed


def log_quiet_telemetry(event_id, source_type, predicted_label, confidence, latency_sec, rating=""):
    """Silently logs anonymous latency & usability ratings on host for evaluation tables."""
    try:
        TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_exists = TELEMETRY_PATH.exists()
        with open(TELEMETRY_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "timestamp",
                        "event_id",
                        "input_source",
                        "predicted_class",
                        "confidence_pct",
                        "latency_sec",
                        "user_rating",
                    ]
                )
            writer.writerow(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    event_id,
                    source_type,
                    predicted_label,
                    f"{confidence * 100:.1f}",
                    f"{latency_sec:.2f}",
                    rating,
                ]
            )
    except Exception:
        pass


@st.cache_data
def generate_soothing_pink_noise(duration_sec: int = 12, sr: int = 16000) -> bytes:
    """Synthesizes a soft, warm pink-noise womb-hum audio clip in memory for caregiver soothing."""
    rng = np.random.default_rng(42)
    n_samples = duration_sec * sr
    white = rng.standard_normal(n_samples)
    fft_vals = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sr)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.power(freqs[1:], 0.65)
    scale *= np.exp(-freqs / 850.0)
    warm = np.fft.irfft(fft_vals * scale, n=n_samples)
    warm = warm / (np.max(np.abs(warm)) + 1e-8) * 0.35

    fade_len = int(0.8 * sr)
    warm[:fade_len] *= np.linspace(0.0, 1.0, fade_len)
    warm[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)

    pcm = np.int16(warm * 32767)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def get_baby_subject() -> str:
    name = st.session_state["baby_name"].strip()
    return name if name else "your baby"


def get_age_care_guidance(cry_class: str, age_group: str, last_fed: str, last_nap: str) -> dict:
    """Returns structured, age-specific caregiver guidance across all 4 developmental stages."""
    subject = get_baby_subject()
    cap_subject = subject[0].upper() + subject[1:] if subject != "your baby" else "Your baby"

    if cry_class == "Hunger":
        by_age = {
            "Newborn: 0–3 months": (
                "Newborns typically feed every 2 to 3 hours (8–12 times in 24 hours) due to small stomach capacity. "
                "Watch for early feeding cues such as rooting, lip-smacking, or bringing hands toward the mouth."
            ),
            "Young Infant: 3–6 months": (
                "Young infants aged 3–6 months usually feed every 3 to 4 hours and may cluster-feed during growth spurts. "
                "Offering a calm, low-distraction feeding space helps prevent distracted unlatching."
            ),
            "Older Infant: 6–12 months": (
                "Older infants combine regular breastmilk or formula feeds with age-appropriate iron-rich purees and soft finger foods. "
                "Check whether it is close to a scheduled milk feed or small meal window."
            ),
            "Young Toddler: 12–18 months": (
                "Young toddlers need regular balanced meals plus 1–2 planned healthy snacks and water/milk sips between meals, "
                "as active play burns energy quickly."
            ),
        }
        return {
            "headline": f"{cap_subject}'s cry may indicate Hunger",
            "context_note": f"Reported last feed: {last_fed}",
            "body": by_age.get(age_group, by_age["Newborn: 0–3 months"]),
            "steps": [
                "Hold gently for 30 seconds to calm rapid breathing before offering a feed.",
                "Offer age-appropriate milk or scheduled snack; pause midway to burp young infants.",
                "Observe fullness cues such as slowed pacing, turning away, or relaxed open hands.",
            ],
        }

    if cry_class == "Pain_Discomfort":
        by_age = {
            "Newborn: 0–3 months": (
                "Trapped air and immature digestion are very common in the first three months. "
                "Gentle over-the-shoulder burping, clockwise tummy massage, and slow bicycle legs often relieve gas pressure."
            ),
            "Young Infant: 3–6 months": (
                "Discomfort in this stage is frequently linked to trapped gas, a damp diaper, tight clothing seams, "
                "room temperature shifts (ideal 20–22°C), or early gum sensitivity."
            ),
            "Older Infant: 6–12 months": (
                "Emerging teeth, digestive gas from new solid foods, or minor physical frustration are common triggers. "
                "Check diaper dryness, clothing comfort, and offer a chilled (not frozen) teething ring if gums look tender."
            ),
            "Young Toddler: 12–18 months": (
                "Molar teething pressure, minor bumps during walking practice, or sensory overstimulation are frequent causes. "
                "Check skin, diaper comfort, and offer calm physical reassurance."
            ),
        }
        return {
            "headline": f"{cap_subject}'s cry suggests Physical Discomfort",
            "context_note": f"Stage: {age_group}",
            "body": by_age.get(age_group, by_age["Newborn: 0–3 months"]),
            "steps": [
                "Burp upright against your shoulder or perform slow clockwise belly strokes and bicycle legs.",
                "Check diaper dryness, room temperature, and snug socks, zippers, or clothing tags.",
                "For teething infants/toddlers, offer a clean, chilled silicone teether for gentle gum relief.",
            ],
        }

    by_age = {
        "Newborn: 0–3 months": (
            "Newborns have short wake windows of 45 to 60 minutes. Staying awake longer triggers overtiredness, "
            "making it harder to drift off without gentle sensory calming."
        ),
        "Young Infant: 3–6 months": (
            "Typical wake windows at 3–6 months last 1.5 to 2.5 hours. Early tiredness signs include yawning, "
            "red eyebrows, rubbing eyes, or turning away from stimulation."
        ),
        "Older Infant: 6–12 months": (
            "Older infants generally stay awake 2.5 to 3.5 hours between naps. "
            "A predictable 10-minute wind-down routine helps signal that sleep time is approaching."
        ),
        "Young Toddler: 12–18 months": (
            "Young toddlers transition toward 1–2 daytime naps with 3.5 to 5 hour wake windows. "
            "Quiet reading, dim lighting, and a consistent bedtime sequence help ease nap resistance."
        ),
    }
    return {
        "headline": f"{cap_subject}'s cry may indicate Sleepiness",
        "context_note": f"Reported last nap: {last_nap}",
        "body": by_age.get(age_group, by_age["Newborn: 0–3 months"]),
        "steps": [
            "Dim bright lights and reduce active play or screen/toy stimulation.",
            "Use a breathable swaddle (0–3m) or sleep sack alongside steady, soft pink noise.",
            "Place baby flat on their back on a firm, clear sleep surface.",
        ],
    }


def render_donut_svg(counts_dict: dict) -> str:
    """Clean SVG donut visualization in BabyCue's Dusty Rose & Mauve palette."""
    total = sum(counts_dict.values())
    if total == 0:
        return ""
    radius = 52
    circumference = 2 * np.pi * radius
    offset = 0.0
    arcs = []
    for key in ["Hunger", "Pain_Discomfort", "Sleepiness"]:
        val = counts_dict.get(key, 0)
        if val <= 0:
            continue
        pct = val / total
        dash = pct * circumference
        arcs.append(
            f'<circle cx="72" cy="72" r="{radius}" fill="transparent" stroke="{CLASS_COLORS[key]}" '
            f'stroke-width="18" stroke-dasharray="{dash:.2f} {circumference:.2f}" '
            f'stroke-dashoffset="-{offset:.2f}" transform="rotate(-90 72 72)"/>'
        )
        offset += dash

    return (
        '<div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap;justify-content:center;padding:8px 0;">'
        '<svg width="144" height="144" viewBox="0 0 144 144">'
        f'<circle cx="72" cy="72" r="{radius}" fill="transparent" stroke="#F4E2E6" stroke-width="18"/>'
        f'{"".join(arcs)}'
        f'<text x="72" y="70" text-anchor="middle" font-family="Plus Jakarta Sans, sans-serif" font-size="20" font-weight="700" fill="#343035">{total}</text>'
        '<text x="72" y="87" text-anchor="middle" font-family="DM Sans, sans-serif" font-size="11" fill="#81777A">Recordings</text>'
        "</svg>"
        '<div style="display:flex;flex-direction:column;gap:9px;font-size:0.9rem;color:#343035;">'
        '<div style="display:flex;align-items:center;gap:8px;">'
        '<span style="width:10px;height:10px;border-radius:50%;background:#C994A2;display:inline-block;"></span>'
        f"<span><b>Hunger:</b> {counts_dict.get('Hunger', 0)} ({counts_dict.get('Hunger', 0)/total*100:.0f}%)</span>"
        "</div>"
        '<div style="display:flex;align-items:center;gap:8px;">'
        '<span style="width:10px;height:10px;border-radius:50%;background:#765A63;display:inline-block;"></span>'
        f"<span><b>Pain / Discomfort:</b> {counts_dict.get('Pain_Discomfort', 0)} ({counts_dict.get('Pain_Discomfort', 0)/total*100:.0f}%)</span>"
        "</div>"
        '<div style="display:flex;align-items:center;gap:8px;">'
        '<span style="width:10px;height:10px;border-radius:50%;background:#DFB0BC;display:inline-block;"></span>'
        f"<span><b>Sleepiness:</b> {counts_dict.get('Sleepiness', 0)} ({counts_dict.get('Sleepiness', 0)/total*100:.0f}%)</span>"
        "</div>"
        "</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Left Sidebar Navigation (Primary + Secondary Nav with Lucide Icons)
# ---------------------------------------------------------------------------
with st.sidebar:
    render_html(
        f"""
        <div style="text-align:center;padding:4px 4px 16px;">
            {babycue_logo_svg(62)}
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:1.38rem;font-weight:800;color:#343035;margin-top:8px;letter-spacing:-0.02em;">
                BabyCue
            </div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.82rem;color:#81777A;margin-top:2px;">
                Every little cry has a cue.
            </div>
        </div>
        """
    )

    for page_name, _ in NAV_ITEMS[:4]:
        btn_type = "primary" if st.session_state["nav"] == page_name else "secondary"
        if st.button(page_name, key=f"side_nav_{page_name}", type=btn_type, use_container_width=True):
            navigate_to(page_name)
            st.rerun()

    render_html("<div style='height:1px;background:#EAD9DD;margin:14px 4px;'></div>")

    for page_name, _ in NAV_ITEMS[4:]:
        btn_type = "primary" if st.session_state["nav"] == page_name else "secondary"
        if st.button(page_name, key=f"side_nav_{page_name}", type=btn_type, use_container_width=True):
            navigate_to(page_name)
            st.rerun()

    active_name = st.session_state["baby_name"].strip() or "Baby Profile"
    bday_sub = ""
    if st.session_state["baby_bday"] is not None:
        exact_age_str, _ = compute_age_from_birthday(st.session_state["baby_bday"])
        bday_sub = f" ({exact_age_str})"

    render_html(
        f"""
        <div style="margin-top:1.25rem;background:#F8F3F1;border:1px solid #EAD9DD;border-radius:16px;padding:12px 14px;">
            <div style="display:flex;align-items:center;gap:8px;font-family:'Plus Jakarta Sans',sans-serif;font-size:0.83rem;font-weight:700;color:#343035;">
                {lucide_svg('user-round', 15, '#765A63')}
                <span>{active_name}</span>
            </div>
            <div style="font-size:0.78rem;color:#81777A;margin-top:3px;padding-left:23px;">
                {st.session_state['baby_age']}{bday_sub}
            </div>
        </div>
        """
    )


# ---------------------------------------------------------------------------
# Top Utility Bar with Interactive Baby Profile Popover (Optional Name + Bday + 4 Categories)
# ---------------------------------------------------------------------------
top_l, top_r = st.columns([2.8, 1.7], vertical_alignment="center")
with top_l:
    render_html(
        f"""
        <div class="bc-pill-badge">
            {lucide_svg('sparkles', 14, '#765A63')}
            <span>AI-Powered Acoustic Analysis</span>
        </div>
        """
    )
with top_r:
    pill_name = st.session_state["baby_name"].strip()
    pill_text = f"{pill_name} • {st.session_state['baby_age']}" if pill_name else st.session_state["baby_age"]
    with st.popover(pill_text, use_container_width=True):
        st.markdown("**Baby Profile (Optional)**")
        st.caption("Personalize care tips by adding a name, birth date, or age stage.")

        pop_name = st.text_input(
            "Baby's Name (Optional)",
            value=st.session_state["baby_name"],
            max_chars=24,
            placeholder="e.g., Maya or Leo",
            key="pop_baby_name",
        )

        use_bday = st.checkbox(
            "Calculate age automatically from birth date",
            value=st.session_state["baby_bday"] is not None,
            key="pop_use_bday",
        )

        if use_bday:
            default_d = st.session_state["baby_bday"] or date.today()
            pop_bday = st.date_input(
                "Baby's Birth Date",
                value=default_d,
                min_value=date(2024, 1, 1),
                max_value=date.today(),
                key="pop_bday_picker",
            )
            exact_str, auto_cat = compute_age_from_birthday(pop_bday)
            st.caption(f"Calculated age: **{exact_str}** → **{auto_cat}**")
            pop_age = auto_cat
        else:
            pop_bday = None
            pop_age = st.selectbox(
                "Select Age Category",
                AGE_CATEGORIES,
                index=AGE_CATEGORIES.index(st.session_state["baby_age"]),
                key="pop_age_cat",
            )

        if st.button("Apply Profile", type="primary", use_container_width=True, key="pop_save_btn"):
            st.session_state["baby_name"] = pop_name.strip()
            st.session_state["baby_bday"] = pop_bday
            st.session_state["baby_age"] = pop_age
            st.rerun()


# ===========================================================================
# PAGE 1: HOME & CRY ANALYZER
# ===========================================================================
if st.session_state["nav"] == "Home":
    subject = get_baby_subject()
    hero_sub = (
        f"Get instant insights into {subject}'s cries and discover gentle, age-aware comfort guidance."
        if subject != "your baby"
        else "Get instant insights into your baby's cries, and find the comfort they need."
    )

    h_col1, h_col2 = st.columns([2.2, 1.0], vertical_alignment="center")
    with h_col1:
        render_html(
            f"""
            <div class="bc-hero-title">
                Because every<br>little cry has a <span class="bc-hero-accent">cue.</span>
            </div>
            <div class="bc-hero-sub">{hero_sub}</div>
            """
        )
    with h_col2:
        render_html(f'<div style="display:flex;justify-content:flex-end;">{decorative_hero_art_svg()}</div>')

    render_html("<div style='height:8px;'></div>")

    if model is None or yamnet_model is None:
        st.error("Trained model artifact not found at 'models/best_model.pkl'. Please run 'python src/train_model.py' first.")
        st.stop()

    main_left, main_right = st.columns([1.65, 1.15], gap="large")

    with main_left:
        render_html(
            f"""
            <div class="bc-card" style="margin-bottom:12px;">
                <div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap;">
                    <div style="width:74px;height:74px;border-radius:50%;background:#F4E2E6;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
                        {lucide_svg('audio-lines', 32, '#765A63', 1.8)}
                    </div>
                    <div style="flex:1;min-width:210px;">
                        <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:1.25rem;font-weight:700;color:#343035;">
                            Upload or Record a Cry
                        </div>
                        <div style="font-size:0.92rem;color:#81777A;margin-top:4px;line-height:1.5;">
                            Let our AI analyze the acoustic pattern (0.8 to 30 seconds) and share what might be going on.
                        </div>
                    </div>
                </div>
            </div>
            """
        )

        ctx1, ctx2 = st.columns(2)
        with ctx1:
            last_fed = st.selectbox(
                "Last Feeding",
                ["Not specified", "Within 1 hour", "1–2 hours ago", "2–3 hours ago", "3+ hours ago"],
                key="ctx_fed",
            )
        with ctx2:
            last_nap = st.selectbox(
                "Last Nap",
                ["Not specified", "Within 1 hour", "1–2 hours ago", "2+ hours ago"],
                key="ctx_nap",
            )

        tab_rec, tab_up = st.tabs(["Record Audio", "Upload File"])
        recorded_clip = None
        uploaded_clip = None
        with tab_rec:
            recorded_clip = st.audio_input(
                "Tap microphone to capture cry audio (1–30 seconds):",
                sample_rate=16000,
                key="mic_recorder",
            )
        with tab_up:
            uploaded_clip = st.file_uploader(
                "Select an audio recording (WAV, MP3, OGG, M4A, 3GP • Max 10 MB):",
                type=["wav", "mp3", "ogg", "m4a", "3gp"],
                key="file_uploader",
            )

        active_audio = uploaded_clip if uploaded_clip is not None else recorded_clip
        source_label = "File Upload" if uploaded_clip is not None else "Live Microphone"

        if active_audio is not None:
            raw_bytes = active_audio.getvalue()
            st.audio(raw_bytes)

            if st.button("Analyze Cry Audio", type="primary", use_container_width=True):
                t_start = time.perf_counter()
                with st.status("Extracting acoustic features with YAMNet AI...", expanded=True) as status:
                    try:
                        file_mb = len(raw_bytes) / (1024 * 1024)
                        if file_mb > MAX_FILE_MB:
                            raise ValueError(f"File size ({file_mb:.1f} MB) exceeds the 10 MB limit.")

                        suffix = Path(active_audio.name).suffix.lower() if hasattr(active_audio, "name") else ".wav"
                        if not suffix:
                            suffix = ".wav"

                        with TemporaryDirectory() as temp_dir:
                            temp_path = Path(temp_dir) / f"cry_input{suffix}"
                            temp_path.write_bytes(raw_bytes)
                            X_live, y_wave = extract_live_yamnet_features(temp_path, yamnet_model)
                            probs = model.predict_proba(X_live)[0]

                        latency = time.perf_counter() - t_start
                        top_idx = int(np.argmax(probs))
                        top_class = str(model.classes_[top_idx])
                        top_prob = float(probs[top_idx])
                        anon_id = f"BC-{random.randint(1000, 9999)}"
                        now_str = datetime.now().strftime("%b %d, %Y • %I:%M %p")

                        step = max(1, len(y_wave) // 180)
                        wave_preview = y_wave[::step].tolist()

                        entry = {
                            "id": anon_id,
                            "timestamp": now_str,
                            "baby_name": st.session_state["baby_name"].strip() or "Baby",
                            "baby_age": st.session_state["baby_age"],
                            "top_class": top_class,
                            "top_prob": top_prob,
                            "probs": {str(c): float(p) for c, p in zip(model.classes_, probs)},
                            "latency": latency,
                            "last_fed": last_fed,
                            "last_nap": last_nap,
                            "wave_preview": wave_preview,
                            "saved": bool(st.session_state["privacy_save"]),
                        }
                        st.session_state["last_result"] = entry
                        if entry["saved"]:
                            st.session_state["cry_history"].insert(0, entry)

                        log_quiet_telemetry(anon_id, source_label, top_class, top_prob, latency, rating="")
                        status.update(label=f"Analysis completed in {latency:.2f}s", state="complete")
                        st.rerun()

                    except Exception as exc:
                        status.update(label="Unable to process audio", state="error")
                        st.error(f"Audio notice: {exc}")

    with main_right:
        latest = st.session_state.get("last_result")
        if latest is not None:
            l_label = LABELS.get(latest["top_class"], latest["top_class"])
            guidance_preview = get_age_care_guidance(
                latest["top_class"], latest["baby_age"], latest["last_fed"], latest["last_nap"]
            )
            render_html(
                f"""
                <div class="bc-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                        <div class="bc-section-label" style="margin:0;">
                            {lucide_svg('audio-lines', 18, '#765A63')}
                            <span>Recent Analysis</span>
                        </div>
                        <span style="font-size:0.8rem;color:#81777A;">{latest['id']}</span>
                    </div>
                    <div style="padding:8px 0 12px;">
                        {waveform_bars_svg(latest.get('wave_preview'), height=48, bar_count=38)}
                    </div>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                        <span class="bc-pill-badge">
                            <span style="width:7px;height:7px;border-radius:50%;background:#C994A2;display:inline-block;"></span>
                            Possible {l_label} ({latest['top_prob']*100:.0f}%)
                        </span>
                        <span style="font-size:0.78rem;color:#81777A;">{latest['timestamp'].split('•')[-1].strip()}</span>
                    </div>
                    <div style="font-size:0.9rem;color:#343035;line-height:1.5;margin-top:6px;">
                        {guidance_preview['headline']}.
                    </div>
                </div>
                """
            )
            if st.button("View Full Insights & History", key="recent_view_hist", use_container_width=True):
                navigate_to("Insights")
                st.rerun()
        else:
            render_html(
                f"""
                <div class="bc-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                        <div class="bc-section-label" style="margin:0;">
                            {lucide_svg('audio-lines', 18, '#765A63')}
                            <span>Recent Analysis</span>
                        </div>
                        <span style="font-size:0.8rem;color:#81777A;">Session-Private</span>
                    </div>
                    <div style="padding:10px 0 14px;opacity:0.65;">
                        {waveform_bars_svg(None, height=44, bar_count=38)}
                    </div>
                    <div style="font-size:0.9rem;color:#81777A;line-height:1.55;">
                        No cry recording analyzed in this session yet. Capture or upload a clip on the left to view live acoustic probabilities and tailored comfort steps.
                    </div>
                </div>
                """
            )

    res = st.session_state.get("last_result")
    if res is not None:
        render_html("<div style='height:8px;'></div>")
        top_class = res["top_class"]
        top_prob = res["top_prob"]
        top_label = LABELS.get(top_class, top_class)
        guide = get_age_care_guidance(top_class, res["baby_age"], res["last_fed"], res["last_nap"])

        if top_prob < 0.50:
            st.warning(
                f"Low Confidence ({top_prob * 100:.1f}%): Acoustic features overlap across categories or contain "
                f"background noise. Please submit a clearer recording closer to your baby. "
                f"Closest acoustic lean: {top_label}."
            )

        res_c1, res_c2 = st.columns([1.25, 1.15], gap="large")
        with res_c1:
            render_html(
                f"""
                <div class="bc-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                        <div class="bc-section-label" style="margin:0;">
                            {lucide_svg('check-circle', 18, '#765A63')}
                            <span>Acoustic Probability Breakdown</span>
                        </div>
                        <span style="font-size:0.8rem;color:#81777A;">Processed in {res['latency']:.2f}s</span>
                    </div>
                </div>
                """
            )
            for cls_name in model.classes_:
                cls_key = str(cls_name)
                p = res["probs"].get(cls_key, 0.0)
                col_a, col_b = st.columns([4, 1], vertical_alignment="center")
                with col_a:
                    st.markdown(f"**{LABELS.get(cls_key, cls_key)}**")
                    st.progress(float(p))
                with col_b:
                    render_html(
                        f"<div style='text-align:right;font-family:\"Plus Jakarta Sans\",sans-serif;font-weight:700;color:#343035;'>{p*100:.1f}%</div>"
                    )

            with st.expander("Inspect Normalized Acoustic Waveform"):
                st.line_chart(pd.DataFrame({"Amplitude": res["wave_preview"]}), height=145, color="#C994A2")

        with res_c2:
            steps_html = "".join(
                [f"<li style='margin-bottom:7px;color:#343035;'>{step_txt}</li>" for step_txt in guide["steps"]]
            )
            render_html(
                f"""
                <div class="bc-card-blush">
                    <div class="bc-pill-badge" style="background:#FFFFFF;margin-bottom:10px;">
                        {lucide_svg('heart', 14, '#765A63')}
                        <span>{guide['context_note']}</span>
                    </div>
                    <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:1.18rem;font-weight:700;color:#343035;margin-bottom:8px;">
                        {guide['headline']}
                    </div>
                    <p style="font-size:0.93rem;color:#343035;line-height:1.58;margin-bottom:12px;">
                        {guide['body']}
                    </p>
                    <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.85rem;font-weight:700;color:#765A63;margin-bottom:6px;">
                        Recommended Comfort Steps
                    </div>
                    <ul style="padding-left:18px;margin:0;font-size:0.9rem;line-height:1.5;">
                        {steps_html}
                    </ul>
                </div>
                """
            )

            rate_l, rate_r = st.columns([1.4, 2.2], vertical_alignment="center")
            with rate_l:
                st.caption("Rate prediction helpfulness:")
            with rate_r:
                if res["id"] in st.session_state["rated_ids"]:
                    st.caption(f"Feedback recorded ({st.session_state['rated_ids'][res['id']]}/5). Thank you.")
                else:
                    r_cols = st.columns(5)
                    for score in range(1, 6):
                        if r_cols[score - 1].button(f"{score}", key=f"rate_{res['id']}_{score}"):
                            st.session_state["rated_ids"][res["id"]] = score
                            log_quiet_telemetry(
                                res["id"], "Usability Rating", top_class, top_prob, res["latency"], rating=f"{score}/5"
                            )
                            st.rerun()

        render_html(
            """
            <div class="bc-disclaimer">
                <b>Non-Clinical Decision-Support Notice:</b> BabyCue estimates acoustic patterns using machine learning and does not
                provide a medical diagnosis. Always rely on caregiver observation and consult a pediatrician if your baby shows fever,
                persistent inconsolable crying, or unusual clinical symptoms.
            </div>
            """
        )

    render_html("<div style='height:14px;'></div>")
    bot_left, bot_right = st.columns([1.65, 1.15], gap="large")

    with bot_left:
        render_html(
            f"""
            <div class="bc-section-label">
                {lucide_svg('lightbulb', 18, '#765A63')}
                <span>How It Works</span>
            </div>
            <div class="bc-card">
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:18px;text-align:center;">
                    <div>
                        <div class="bc-step-circle">{lucide_svg('audio-lines', 24, '#765A63')}</div>
                        <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:0.96rem;color:#343035;">
                            1. Record
                        </div>
                        <div style="font-size:0.84rem;color:#81777A;margin-top:4px;line-height:1.45;">
                            Capture your baby's cry or upload an existing audio clip.
                        </div>
                    </div>
                    <div>
                        <div class="bc-step-circle">{lucide_svg('brain', 24, '#765A63')}</div>
                        <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:0.96rem;color:#343035;">
                            2. Analyze
                        </div>
                        <div style="font-size:0.84rem;color:#81777A;margin-top:4px;line-height:1.45;">
                            Our AI detects acoustic patterns and identifies likely needs.
                        </div>
                    </div>
                    <div>
                        <div class="bc-step-circle">{lucide_svg('heart', 24, '#765A63')}</div>
                        <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:0.96rem;color:#343035;">
                            3. Understand
                        </div>
                        <div style="font-size:0.84rem;color:#81777A;margin-top:4px;line-height:1.45;">
                            Receive clear probabilities and age-specific comfort guidance.
                        </div>
                    </div>
                </div>
            </div>
            """
        )

    with bot_right:
        render_html(
            f"""
            <div class="bc-card-blush" style="position:relative;overflow:hidden;">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;">
                    <div>
                        <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:1.15rem;font-weight:700;color:#343035;">
                            Gentle Care. Smarter Support.
                        </div>
                        <div style="font-size:0.88rem;color:#5D4A50;margin-top:6px;line-height:1.5;">
                            From cry analysis to age-specific soothing guides, we are here to support you every step of the way.
                        </div>
                    </div>
                    <div style="flex-shrink:0;">{bottle_corner_svg()}</div>
                </div>
            </div>
            """
        )
        if st.button("Explore Care Guides", type="primary", use_container_width=True):
            navigate_to("Care Guides")
            st.rerun()


# ===========================================================================
# PAGE 2: INSIGHTS & CRY HISTORY
# ===========================================================================
elif st.session_state["nav"] == "Insights":
    subject = get_baby_subject()
    title_prefix = f"{subject[0].upper() + subject[1:]}'s" if subject != "your baby" else "Session"
    st.markdown(f"## {title_prefix} Insights & Cry History")
    st.caption(
        "All entries below are stored exclusively in your private browser session and are never visible to other visitors."
    )

    history = st.session_state["cry_history"]

    if not history:
        render_html(
            f"""
            <div class="bc-card" style="text-align:center;padding:42px 24px;">
                <div class="bc-step-circle" style="width:64px;height:64px;">
                    {lucide_svg('history', 26, '#765A63')}
                </div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:1.15rem;font-weight:700;color:#343035;margin-top:8px;">
                    No Saved Cry Recordings Yet
                </div>
                <p style="color:#81777A;max-width:460px;margin:8px auto 0;font-size:0.93rem;line-height:1.55;">
                    Record or upload a cry clip on the Home screen with session journaling enabled to track patterns, confidence trends, and timestamps here.
                </p>
            </div>
            """
        )
        if st.button("Go to Cry Analyzer", type="primary"):
            navigate_to("Home")
            st.rerun()
    else:
        counts = {"Hunger": 0, "Pain_Discomfort": 0, "Sleepiness": 0}
        for item in history:
            counts[item["top_class"]] = counts.get(item["top_class"], 0) + 1

        top_cat = max(counts, key=counts.get)
        avg_conf = float(np.mean([x["top_prob"] for x in history])) * 100
        avg_lat = float(np.mean([x["latency"] for x in history]))

        k1, k2, k3, k4 = st.columns(4, gap="medium")
        with k1:
            render_html(
                f"""
                <div class="bc-kpi-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span class="bc-kpi-lbl">Analyzed Cries</span>
                        {lucide_svg('audio-lines', 16, '#765A63')}
                    </div>
                    <div class="bc-kpi-val">{len(history)}</div>
                    <div class="bc-kpi-lbl">Current private session</div>
                </div>
                """
            )
        with k2:
            render_html(
                f"""
                <div class="bc-kpi-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span class="bc-kpi-lbl">Primary Pattern</span>
                        {lucide_svg('heart', 16, '#765A63')}
                    </div>
                    <div class="bc-kpi-val" style="font-size:1.3rem;">{LABELS[top_cat]}</div>
                    <div class="bc-kpi-lbl">{counts[top_cat]} of {len(history)} recordings</div>
                </div>
                """
            )
        with k3:
            render_html(
                f"""
                <div class="bc-kpi-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span class="bc-kpi-lbl">Mean Confidence</span>
                        {lucide_svg('chart', 16, '#765A63')}
                    </div>
                    <div class="bc-kpi-val">{avg_conf:.1f}%</div>
                    <div class="bc-kpi-lbl">Across logged clips</div>
                </div>
                """
            )
        with k4:
            render_html(
                f"""
                <div class="bc-kpi-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span class="bc-kpi-lbl">Avg. Latency</span>
                        {lucide_svg('clock', 16, '#765A63')}
                    </div>
                    <div class="bc-kpi-val">{avg_lat:.2f}s</div>
                    <div class="bc-kpi-lbl">Real-time inference</div>
                </div>
                """
            )

        render_html("<div style='height:14px;'></div>")

        c_left, c_right = st.columns(2, gap="large")
        with c_left:
            render_html(
                f"""
                <div class="bc-card">
                    <div class="bc-section-label">
                        {lucide_svg('chart', 18, '#765A63')}
                        <span>Cry Category Distribution</span>
                    </div>
                    {render_donut_svg(counts)}
                </div>
                """
            )

        with c_right:
            render_html(
                f"""
                <div class="bc-card">
                    <div class="bc-section-label">
                        {lucide_svg('audio-lines', 18, '#765A63')}
                        <span>Confidence Trajectory</span>
                    </div>
                </div>
                """
            )
            df_trend = pd.DataFrame(
                {
                    "Clip": [f"Clip {i + 1}" for i in range(len(history))],
                    "Confidence (%)": [round(item["top_prob"] * 100, 1) for item in reversed(history)],
                }
            ).set_index("Clip")
            st.bar_chart(df_trend, color="#C994A2", height=168)

        act1, act2, act3 = st.columns([2.2, 1.1, 1.0], vertical_alignment="center")
        with act1:
            st.markdown("### Recorded Cry Timeline")
        with act2:
            summary_lines = [
                f"BabyCue Session Summary — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"Baby Profile: {st.session_state['baby_name'].strip() or 'Unnamed'} ({st.session_state['baby_age']})",
                f"Total Recordings: {len(history)} | Dominant Pattern: {LABELS[top_cat]}",
                "-" * 56,
            ]
            for item in history:
                summary_lines.append(
                    f"[{item['timestamp']}] {item['id']} — {LABELS[item['top_class']]} "
                    f"({item['top_prob']*100:.1f}% confidence) | Fed: {item['last_fed']} | Nap: {item['last_nap']}"
                )
            st.download_button(
                "Export Session Summary",
                data="\n".join(summary_lines),
                file_name="babycue_session_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with act3:
            if st.button("Clear History", use_container_width=True):
                st.session_state["cry_history"] = []
                st.rerun()

        for idx, item in enumerate(history):
            cat_label = LABELS.get(item["top_class"], item["top_class"])
            render_html(
                f"""
                <div class="bc-card" style="padding:16px 20px;margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
                        <div style="min-width:190px;">
                            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:1rem;color:#343035;">
                                {cat_label}
                                <span style="font-size:0.78rem;font-weight:600;color:#765A63;background:#F4E2E6;padding:2px 9px;border-radius:999px;margin-left:6px;">
                                    {item['id']}
                                </span>
                            </div>
                            <div style="font-size:0.82rem;color:#81777A;margin-top:3px;">
                                {item['timestamp']} • {item['baby_age']}
                            </div>
                        </div>
                        <div style="flex:1;min-width:150px;max-width:260px;">
                            {waveform_bars_svg(item.get('wave_preview'), height=32, bar_count=32)}
                        </div>
                        <div style="text-align:right;">
                            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;color:#343035;">
                                {item['top_prob']*100:.1f}% confidence
                            </div>
                            <div style="font-size:0.78rem;color:#81777A;">
                                Latency: {item['latency']:.2f}s
                            </div>
                        </div>
                    </div>
                </div>
                """
            )
            _, d_col2 = st.columns([4.2, 0.8])
            with d_col2:
                if st.button("Delete", key=f"del_hist_{item['id']}_{idx}", use_container_width=True):
                    st.session_state["cry_history"].pop(idx)
                    st.rerun()


# ===========================================================================
# PAGE 3: CARE GUIDES & ROUTINE
# ===========================================================================
elif st.session_state["nav"] == "Care Guides":
    st.markdown("## Care Guides & Daily Routine")
    st.caption("Evidence-informed caregiver guidance and soothing routines tailored to developmental stage.")

    selected_age = st.selectbox(
        "Viewing developmental guidance for:",
        AGE_CATEGORIES,
        index=AGE_CATEGORIES.index(st.session_state["baby_age"]),
        key="guides_age_selector",
    )
    if selected_age != st.session_state["baby_age"]:
        st.session_state["baby_age"] = selected_age
        st.rerun()

    st.markdown("### Acoustic Cry Reference")
    e1, e2, e3 = st.columns(3, gap="medium")
    with e1:
        render_html(
            f"""
            <div class="bc-card">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                    <div style="width:36px;height:36px;border-radius:10px;background:#F4E2E6;display:flex;align-items:center;justify-content:center;">
                        {lucide_svg('baby-bottle', 18, '#765A63')}
                    </div>
                    <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:1.02rem;color:#343035;">
                        Hunger Cry
                    </div>
                </div>
                <p style="font-size:0.89rem;color:#343035;line-height:1.55;margin-bottom:8px;">
                    <b>Acoustic Signature:</b> Rhythmic, repetitive rising-and-falling melody around 400–550 Hz with short inspiratory breaths.
                </p>
                <p style="font-size:0.86rem;color:#81777A;line-height:1.5;margin:0;">
                    <b>Early Physical Cues:</b> Rooting toward chest, hand-to-mouth movements, lip smacking, and clenched fists.
                </p>
            </div>
            """
        )
    with e2:
        render_html(
            f"""
            <div class="bc-card">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                    <div style="width:36px;height:36px;border-radius:10px;background:#F4E2E6;display:flex;align-items:center;justify-content:center;">
                        {lucide_svg('heart', 18, '#765A63')}
                    </div>
                    <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:1.02rem;color:#343035;">
                        Pain / Discomfort
                    </div>
                </div>
                <p style="font-size:0.89rem;color:#343035;line-height:1.55;margin-bottom:8px;">
                    <b>Acoustic Signature:</b> Sudden, high-pitched, tense vocal onset with elevated energy and irregular breath-holding pauses.
                </p>
                <p style="font-size:0.86rem;color:#81777A;line-height:1.5;margin:0;">
                    <b>Early Physical Cues:</b> Drawing knees toward the abdomen (gas), arching back, squirming, or facial grimacing.
                </p>
            </div>
            """
        )
    with e3:
        render_html(
            f"""
            <div class="bc-card">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                    <div style="width:36px;height:36px;border-radius:10px;background:#F4E2E6;display:flex;align-items:center;justify-content:center;">
                        {lucide_svg('moon', 18, '#765A63')}
                    </div>
                    <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:1.02rem;color:#343035;">
                        Sleepiness Cry
                    </div>
                </div>
                <p style="font-size:0.89rem;color:#343035;line-height:1.55;margin-bottom:8px;">
                    <b>Acoustic Signature:</b> Breathy, drawn-out, lower-intensity fussing or whining that gradually rises and fades.
                </p>
                <p style="font-size:0.86rem;color:#81777A;line-height:1.5;margin:0;">
                    <b>Early Physical Cues:</b> Yawning, rubbing eyes or ears, reddened eyebrows, and averting gaze from bright objects.
                </p>
            </div>
            """
        )

    st.markdown(f"### Soothing & Care Protocols ({selected_age})")
    art_tab1, art_tab2, art_tab3 = st.tabs(
        ["Feeding & Nutrition Cues", "Gas & Discomfort Relief", "Sleep Windows & Wind-Down"]
    )

    with art_tab1:
        if selected_age == "Newborn: 0–3 months":
            st.info(
                "**Feeding Frequency:** Newborns typically feed on demand every **2 to 3 hours** (roughly 8 to 12 feeds per 24 hours) due to small stomach volume.\n\n"
                "**Recommended Nutrition:** Exclusive breastmilk or iron-fortified infant formula. Avoid water, juice, or solid foods during the first months unless specifically directed by a pediatrician.\n\n"
                "**Comfort Pacing:** Pause midway through feeds to burp upright for 2–3 minutes so swallowed air does not cause post-feed discomfort."
            )
        elif selected_age == "Young Infant: 3–6 months":
            st.info(
                "**Feeding Frequency:** Typically every **3 to 4 hours** (5 to 6 milk feeds daily), with temporary cluster feeding during developmental growth spurts.\n\n"
                "**Recommended Nutrition:** Breastmilk or infant formula remains the primary source of nutrition. Near 6 months—when showing head stability and interest—pediatricians often introduce single-ingredient iron-rich purees.\n\n"
                "**Environment Tip:** Babies at this stage become alert to sounds and movement; feeding in a quiet, softly lit room helps prevent distracted unlatching."
            )
        elif selected_age == "Older Infant: 6–12 months":
            st.info(
                "**Feeding Frequency:** Usually **3 small solid meals and 1–2 snacks** alongside scheduled breastmilk or formula feeds.\n\n"
                "**Recommended Nutrition:** Soft mashed fruits, vegetables, lentils, oatmeal, yogurt, or age-safe finger foods, paired with small sips of water during meals.\n\n"
                "**Mealtime Cue:** Crying shortly before mealtime often signals thirst or hunger—offering milk or water first helps calm a hungry infant before solids."
            )
        else:
            st.info(
                "**Feeding Frequency:** Young toddlers (12–18 months) thrive on **3 family-style small meals and 2 scheduled snacks** daily.\n\n"
                "**Recommended Nutrition:** Nutrient-dense soft table foods (fruits, vegetables, proteins, whole grains) paired with water and whole milk in an open or straw cup.\n\n"
                "**Toddler Cue:** Pre-meal fussiness often stems from low blood sugar after active play—a predictable snack schedule prevents overtired hunger meltdowns."
            )

    with art_tab2:
        st.info(
            "1. **Two-Position Burping:** Hold your baby upright against your chest with their chin resting gently over your shoulder, or sit them on your lap supporting their chin and gently rub upward on their back for 3 to 5 minutes.\n"
            "2. **Clockwise Tummy Massage:** Using warm hands, stroke gently in a **clockwise circle** around the navel—following the natural path of digestion—to help release trapped gas bubbles.\n"
            "3. **Gentle Bicycle Legs:** Lay your baby on their back and slowly move their legs in a smooth cycling motion, then gently bend both knees toward the tummy for 3 seconds.\n"
            "4. **Physical Comfort & Teething Check:** Inspect for a wet diaper, snug elastic bands, hair wrapped around fingers or toes, or emerging gums (offer a chilled silicone teething ring for babies 4+ months).\n"
            "5. **When to Seek Medical Care:** Contact a healthcare professional promptly if crying is accompanied by a fever of 38.0°C (100.4°F) or higher, persistent vomiting, refusal to feed, or unusual lethargy."
        )

    with art_tab3:
        wake_window = {
            "Newborn: 0–3 months": "45 to 60 minutes of total awake time (including feeding)",
            "Young Infant: 3–6 months": "1.5 to 2.5 hours of awake time between naps",
            "Older Infant: 6–12 months": "2.5 to 3.5 hours of awake time between naps",
            "Young Toddler: 12–18 months": "3.5 to 5 hours of awake time (1 to 2 daytime naps)",
        }[selected_age]
        st.info(
            f"**Recommended Wake Window ({selected_age}):** {wake_window}. Starting a wind-down routine near the end of this window helps prevent overtiredness.\n\n"
            "**Five-Minute Wind-Down Sequence:** Dim room lighting, dress baby in a breathable swaddle (0–3m) or sleep sack (3+ months), turn on continuous low-frequency soothing sound, and rock gently with slow rhythmic shushing.\n\n"
            "**Safe Sleep Practice:** Always place your baby flat on their back on a firm, clear sleep surface free of loose blankets, pillows, or soft toys."
        )

    r_left, r_right = st.columns([1.55, 1.05], gap="large")

    with r_left:
        subject = get_baby_subject()
        routine_title = f"{subject[0].upper() + subject[1:]}'s Daily Routine" if subject != "your baby" else "Daily Care Routine"
        st.markdown(f"### {routine_title}")

        for idx, item in enumerate(st.session_state["routine_items"]):
            c1, c2, c3 = st.columns([0.45, 3.2, 1.3], vertical_alignment="center")
            with c1:
                done = st.checkbox("Done", value=item["done"], key=f"rt_{idx}", label_visibility="collapsed")
                st.session_state["routine_items"][idx]["done"] = done
            with c2:
                txt_style = (
                    "text-decoration:line-through;color:#81777A;"
                    if done
                    else "font-weight:600;color:#343035;"
                )
                render_html(f"<span style='{txt_style}'>{item['task']}</span>")
            with c3:
                st.caption(f"{item['time']} • {item['category']}")

        add1, add2, add3 = st.columns([2.2, 1.1, 1.0])
        with add1:
            new_task = st.text_input("Task", placeholder="Add care activity (e.g., Tummy time)", label_visibility="collapsed")
        with add2:
            new_time = st.text_input("Time", placeholder="03:30 PM", label_visibility="collapsed")
        with add3:
            if st.button("Add Item", use_container_width=True) and new_task.strip():
                st.session_state["routine_items"].append(
                    {"task": new_task.strip(), "time": new_time.strip() or "Today", "category": "Care", "done": False}
                )
                st.rerun()

    with r_right:
        render_html(
            f"""
            <div class="bc-card-blush">
                <div class="bc-section-label">
                    {lucide_svg('volume-2', 18, '#765A63')}
                    <span>Calming Sound Companion</span>
                </div>
                <p style="font-size:0.88rem;color:#5D4A50;line-height:1.5;margin-bottom:0;">
                    Play warm, low-frequency pink noise to recreate a gentle womb-like acoustic environment during soothing or nap wind-down.
                </p>
            </div>
            """
        )
        st.audio(generate_soothing_pink_noise(), format="audio/wav")


# ===========================================================================
# PAGE 4: PARENTING FAQ & AI SUPPORT
# ===========================================================================
elif st.session_state["nav"] == "Parenting FAQ":
    st.markdown("## Parenting FAQ & AI Support")
    st.caption("Ask everyday infant care questions or explore evidence-informed answers for new caregivers.")

    faqs = {
        "Why do hunger and discomfort cries sometimes sound similar?": (
            "In early infancy, hunger and mild physical discomfort share overlapping acoustic frequencies between 400 and 600 Hz. "
            "As hunger intensifies, vocal tension increases and begins resembling discomfort. Combining BabyCue's acoustic probabilities "
            "with your Last Feeding context provides a much clearer picture."
        ),
        "How much crying is normal during the first three months?": (
            "Healthy newborns commonly cry 1 to 3 hours per day, peaking around 6 to 8 weeks of age—often in the late afternoon "
            "or evening—before gradually settling by 3 to 4 months. If you ever feel overwhelmed, place your baby safely on their back "
            "in the crib and take a brief breather."
        ),
        "What should I do when BabyCue displays a Low Confidence notice?": (
            "A Low Confidence (< 50%) alert indicates that ambient background noise (such as television, conversation, or fans) "
            "or mixed vocal cues made the recording ambiguous. Try capturing a clearer 3 to 10 second clip closer to your baby in a quieter room."
        ),
        "How is my baby's audio recording protected?": (
            "Audio recordings are processed in ephemeral temporary memory and deleted immediately after feature extraction. "
            "Your session history stays strictly isolated in your own browser session and is never shared across users."
        ),
    }

    for msg in st.session_state["chat_messages"]:
        if msg["role"] == "user":
            render_html(f'<div class="bc-chat-user">{msg["text"]}</div>')
        else:
            render_html(f'<div class="bc-chat-bot">{msg["text"]}</div>')

    st.markdown("#### Suggested Topics")
    q_cols = st.columns(2)
    for idx, (question, answer) in enumerate(faqs.items()):
        if q_cols[idx % 2].button(question, key=f"faq_q_{idx}", use_container_width=True):
            st.session_state["chat_messages"].append({"role": "user", "text": question})
            st.session_state["chat_messages"].append({"role": "assistant", "text": answer})
            st.rerun()

    user_prompt = st.chat_input("Ask about feeding intervals, trapped gas relief, sleep windows, or model accuracy...")
    if user_prompt:
        st.session_state["chat_messages"].append({"role": "user", "text": user_prompt})
        q_low = user_prompt.lower()
        age_now = st.session_state["baby_age"]
        if any(k in q_low for k in ["feed", "milk", "hungry", "hunger", "eat", "bottle"]):
            g = get_age_care_guidance("Hunger", age_now, "Not specified", "Not specified")
            reply = f"{g['headline']}: {g['body']} Key tip: {g['steps'][0]}"
        elif any(k in q_low for k in ["gas", "burp", "pain", "colic", "tummy", "belly", "diaper", "discomfort", "teeth"]):
            g = get_age_care_guidance("Pain_Discomfort", age_now, "Not specified", "Not specified")
            reply = f"{g['headline']}: {g['body']} Key tip: {g['steps'][0]}"
        elif any(k in q_low for k in ["sleep", "nap", "tired", "wake", "night", "rest"]):
            g = get_age_care_guidance("Sleepiness", age_now, "Not specified", "Not specified")
            reply = f"{g['headline']}: {g['body']} Key tip: {g['steps'][0]}"
        else:
            reply = (
                "You can explore detailed age-specific feeding, gas relief, and sleep wind-down checklists in the "
                "Care Guides tab, or select any suggested FAQ above. Reminder: BabyCue offers non-clinical caregiver "
                "support and does not replace professional pediatric advice."
            )
        st.session_state["chat_messages"].append({"role": "assistant", "text": reply})
        st.rerun()

    render_html("<div style='height:12px;'></div>")
    with st.expander("System Architecture & Academic Evaluation Summary"):
        st.markdown(
            """
            * **Acoustic Feature Extraction:** Google's pre-trained **YAMNet** neural network extracts **2,048 deep acoustic embeddings** (1,024 mean-pooled + 1,024 max-pooled) from standardized 16 kHz mono infant cry audio.
            * **Classification Pipeline:** L2 Normalization, StandardScaler, **64-component PCA**, and a **Class-Balanced Logistic Regression** classifier trained on **638 MD5-deduplicated recordings** across `Hunger`, `Pain_Discomfort`, and `Sleepiness`.
            * **Audited 5-Fold Cross-Validation:** Achieves **67.55%** unseen accuracy across all 638 multi-format clips and **69.45%** unseen accuracy across 586 high-SNR WAV recordings.
            * **Safety & Privacy Safeguards:** Enforces input validation (0.8–30s duration, max 10 MB), sub-5-second latency, `< 0.50` low-confidence alerts, and immediate deletion of temporary audio files.
            """
        )


# ===========================================================================
# PAGE 5 & 6 (SECONDARY NAVIGATION): SETTINGS & PROFILE
# ===========================================================================
else:
    st.markdown("## Baby Profile & Privacy Settings")
    st.caption("Personalize care recommendations and manage how your current session stores analysis history.")

    p_col1, p_col2 = st.columns(2, gap="large")
    with p_col1:
        render_html(
            f"""
            <div class="bc-card">
                <div class="bc-section-label">
                    {lucide_svg('user-round', 18, '#765A63')}
                    <span>Baby Profile</span>
                </div>
                <p style="font-size:0.88rem;color:#81777A;line-height:1.5;margin-bottom:0;">
                    Adding your baby's name and birth date is optional. You can also select any developmental age stage directly.
                </p>
            </div>
            """
        )
        name_input = st.text_input(
            "Baby's Name (Optional)",
            value=st.session_state["baby_name"],
            max_chars=24,
            placeholder="e.g., Maya or Leo",
        )
        use_bday_settings = st.checkbox(
            "Use baby's birth date to calculate age automatically",
            value=st.session_state["baby_bday"] is not None,
            key="settings_use_bday",
        )
        if use_bday_settings:
            default_bday_s = st.session_state["baby_bday"] or date.today()
            bday_input = st.date_input(
                "Baby's Birth Date",
                value=default_bday_s,
                min_value=date(2024, 1, 1),
                max_value=date.today(),
                key="settings_bday_picker",
            )
            exact_s, age_input = compute_age_from_birthday(bday_input)
            st.info(f"Calculated age: **{exact_s}** ({age_input})")
        else:
            bday_input = None
            age_input = st.selectbox(
                "Developmental Age Category",
                AGE_CATEGORIES,
                index=AGE_CATEGORIES.index(st.session_state["baby_age"]),
            )

        if st.button("Save Profile Preferences", type="primary"):
            st.session_state["baby_name"] = name_input.strip()
            st.session_state["baby_bday"] = bday_input
            st.session_state["baby_age"] = age_input
            st.success("Profile preferences updated for this session.")

    with p_col2:
        render_html(
            f"""
            <div class="bc-card">
                <div class="bc-section-label">
                    {lucide_svg('shield-check', 18, '#765A63')}
                    <span>Privacy & Session Controls</span>
                </div>
                <p style="font-size:0.88rem;color:#81777A;line-height:1.5;margin-bottom:0;">
                    Raw audio is always processed in temporary memory and deleted immediately after analysis. Choose whether anonymized prediction summaries are kept in your current browser tab.
                </p>
            </div>
            """
        )
        save_choice = st.toggle(
            "Keep anonymized prediction summaries in my private session journal",
            value=st.session_state["privacy_save"],
        )
        st.session_state["privacy_save"] = save_choice

        render_html("<div style='height:10px;'></div>")
        if st.button("Reset All Session Data", use_container_width=True):
            st.session_state["cry_history"] = []
            st.session_state["last_result"] = None
            st.session_state["rated_ids"] = {}
            st.session_state["baby_name"] = ""
            st.session_state["baby_bday"] = None
            st.success("All session history and profile fields have been cleared.")
            

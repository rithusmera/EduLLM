import streamlit as st

# ── Preset themes ────────────────────────────────────
PRESETS = {
    "Purple": "#7c3aed",
    "Blue":   "#3b82f6",
    "Cyan":   "#06b6d4",
    "Green":  "#22c55e",
    "Orange": "#f97316",
    "Rose":   "#f43f5e",
}
DEFAULT_COLOR = "#7c3aed"

# ── Color helpers ─────────────────────────────────────
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)

def _darken(h, amt=28):
    r,g,b = _hex_to_rgb(h)
    return "#{:02x}{:02x}{:02x}".format(max(0,r-amt), max(0,g-amt), max(0,b-amt))

def get_accent():
    return st.session_state.get("accent_color", DEFAULT_COLOR)

# ── Sidebar theme picker ──────────────────────────────
def render_theme_picker():
    current = get_accent()
    st.markdown('<div class="section-label">🎨 THEME</div>', unsafe_allow_html=True)

    # Preset swatch row
    swatch_html = '<div style="display:flex;gap:7px;flex-wrap:wrap;margin-bottom:10px;">'
    for name, color in PRESETS.items():
        border = "3px solid #fff" if color.lower() == current.lower() else "2px solid transparent"
        swatch_html += (
            f'<div title="{name}" style="width:22px;height:22px;border-radius:50%;'
            f'background:{color};border:{border};cursor:pointer;box-sizing:border-box;'
            f'box-shadow:0 2px 6px rgba(0,0,0,0.35);"></div>'
        )
    swatch_html += '</div>'
    st.markdown(swatch_html, unsafe_allow_html=True)

    # One button per preset  (below swatches for actual interaction)
    cols = st.columns(3)
    names = list(PRESETS.keys())
    for i, name in enumerate(names):
        with cols[i % 3]:
            label = f"✓ {name}" if PRESETS[name].lower() == current.lower() else name
            if st.button(label, key=f"t_{name}", use_container_width=True):
                st.session_state.accent_color = PRESETS[name]
                st.rerun()

    # Custom color picker
    st.markdown('<div style="margin-top:6px;font-size:12px;color:#4a5070;">Custom color</div>', unsafe_allow_html=True)
    custom = st.color_picker("", value=current, key="custom_cp", label_visibility="collapsed")
    if custom.lower() != current.lower():
        st.session_state.accent_color = custom
        st.rerun()

# ── Full dark CSS (accent-aware) ──────────────────────
def get_css(extra=""):
    a  = get_accent()
    d  = _darken(a, 22)
    dk = _darken(a, 44)
    r,g,b = _hex_to_rgb(a)
    rgb = f"{r},{g},{b}"

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

/* ── Bases ── */
.stApp {{ background-color: #13141f !important; }}
[data-testid="stSidebarNav"] {{ display: none; }}
#MainMenu {{ visibility: hidden; }}
header    {{ visibility: hidden; }}
footer    {{ visibility: hidden; }}

.stMarkdown, .stText, p, span, label, div, h1, h2, h3 {{ color: #e2e8f0 !important; }}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background-color: #1c1d2e !important;
    border-right: 1px solid #2a2b3d !important;
}}
section[data-testid="stSidebar"] * {{ color: #a0a8c0 !important; }}

/* ── Block container ── */
.block-container {{ padding-top: 2rem !important; }}

/* ── Cards ── */
.edu-card {{
    background-color: #1e2030 !important;
    padding: 1.8rem;
    border-radius: 16px;
    border: 1px solid #2a2b3d;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    margin-bottom: 1.5rem;
}}

/* ── Answer / response box ── */
.answer-box {{
    background-color: #1e2030 !important;
    padding: 1.8rem;
    border-radius: 14px;
    border: 1px solid #2a2b3d;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    line-height: 1.7;
    color: #c8d0e8 !important;
    margin-top: 1rem;
}}
.answer-box * {{ color: #c8d0e8 !important; }}
.katex, .katex * {{ color: #c8d0e8 !important; }}
[data-testid="stLatex"], [data-testid="stLatex"] * {{ color: #c8d0e8 !important; }}

/* ── Placeholder ── */
.placeholder-box {{
    border: 2px dashed #2a2b3d;
    border-radius: 14px;
    padding: 3.5rem 2rem;
    text-align: center;
    color: #4a5070 !important;
    font-weight: 500;
    background-color: #181928 !important;
    margin-top: 1rem;
}}

/* ── Badge ── */
.badge {{
    display: inline-block;
    padding: 4px 12px;
    background-color: rgba({rgb},0.18) !important;
    border-radius: 6px;
    font-size: 11px;
    color: {a} !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
}}

/* ── Sidebar typography ── */
.logo-text {{ font-size: 22px; font-weight: 700; color: #e2e8f0 !important; letter-spacing: -0.025em; }}
.logo-sub  {{ font-size: 12px; color: #6b7280 !important; margin-top: 2px; }}
.sidebar-footer {{
    padding: 1rem 0.5rem;
    border-top: 1px solid #2a2b3d;
    margin-top: 1.5rem;
    font-size: 11px;
    color: #4a5070 !important;
}}
.section-label {{
    font-size: 10px; font-weight: 700; color: #4a5070 !important;
    text-transform: uppercase; letter-spacing: 0.09em;
    margin-bottom: 8px; margin-top: 16px; padding: 0 2px;
}}
.nav-item {{
    display: flex; align-items: center; gap: 10px;
    padding: 8px 10px; border-radius: 8px;
    font-size: 14px; font-weight: 500;
    color: #a0a8c0 !important; margin-bottom: 4px;
}}
.nav-item.active {{
    background-color: rgba({rgb},0.12) !important;
    color: {a} !important;
}}
.nav-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}

/* ── Page header ── */
.page-header {{
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 2rem; padding-bottom: 1.2rem; border-bottom: 1px solid #2a2b3d;
}}
.page-title {{ font-size: 24px; font-weight: 700; color: #e2e8f0 !important; }}
.page-sub   {{ font-size: 13px; color: #6b7280 !important; margin-top: 3px; }}
.rag-badge {{
    background: rgba({rgb},0.14);
    border: 1px solid rgba({rgb},0.3);
    color: {a} !important;
    padding: 5px 14px; border-radius: 20px;
    font-size: 12px; font-weight: 600;
}}

/* ── Textarea ── */
.stTextArea > div > div > textarea {{
    background-color: #1e2030 !important;
    border: 1px solid #2a2b3d !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 14px !important;
}}
.stTextArea > div > div > textarea:focus {{
    border-color: {a} !important;
    box-shadow: 0 0 0 3px rgba({rgb},0.22) !important;
}}
.stTextArea > div > div > textarea::placeholder {{ color: #4a5070 !important; }}

/* ── Text input (login) ── */
.stTextInput > div > div > input {{
    background-color: #13141f !important;
    border: 1px solid #2a2b3d !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    height: 2.8rem !important;
    font-size: 14px !important;
}}
.stTextInput > div > div > input:focus {{
    border-color: {a} !important;
    box-shadow: 0 0 0 3px rgba({rgb},0.22) !important;
}}
.stTextInput > div > div > input::placeholder {{ color: #4a5070 !important; }}

/* ── Primary button (global) ── */
div.stButton > button {{
    background: linear-gradient(135deg, {a}, {d}) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 0.55rem 1.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 10px rgba({rgb},0.28) !important;
}}
div.stButton > button:hover {{
    background: linear-gradient(135deg, {d}, {dk}) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 18px rgba({rgb},0.42) !important;
    color: white !important;
}}

/* ── Chip / suggestion buttons (columns) ── */
div[data-testid="stColumn"] button {{
    background-color: #1e2030 !important;
    color: #a0a8c0 !important;
    border: 1px solid #2a2b3d !important;
    padding: 0.35rem 0.5rem !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    border-radius: 20px !important;
    white-space: nowrap !important;
    box-shadow: none !important;
    transform: none !important;
}}
div[data-testid="stColumn"] button:hover {{
    border-color: {a} !important;
    color: {a} !important;
    background-color: rgba({rgb},0.08) !important;
    transform: none !important;
    box-shadow: none !important;
}}

/* ── Radio & checkbox ── */
.stRadio label, .stCheckbox label {{ color: #a0a8c0 !important; }}

/* ── Selectbox ── */
.stSelectbox > div > div {{
    background-color: #1e2030 !important;
    border: 1px solid #2a2b3d !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}}

/* ── Tabs (login) ── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: #13141f !important;
    border-radius: 10px;
    gap: 3px; padding: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent !important;
    color: #6b7280 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    flex: 1; justify-content: center;
}}
.stTabs [aria-selected="true"] {{
    background-color: #2a2b3d !important;
    color: #e2e8f0 !important;
}}

/* ── Divider ── */
hr {{
    border: 0 !important;
    border-top: 1px solid #2a2b3d !important;
    margin: 1.2rem 0 !important;
}}

/* ── page_link ── */
[data-testid="stPageLink"] a {{
    color: #a0a8c0 !important;
    text-decoration: none !important;
    font-size: 14px !important;
}}
[data-testid="stPageLink"] a:hover {{ color: #e2e8f0 !important; }}

{extra}
</style>
"""

"""
shared_theme.py
---------------
Shared CSS theme and flag utilities for multi-page Streamlit app.
Imported by both the main predictor page and the group stage simulator.
"""

import streamlit as st


# ══════════════════════════════════════════════════════════════════════════════
# CSS Theme — Tactical Analytics Design System
# ══════════════════════════════════════════════════════════════════════════════

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

/* Global Dynamic Theme System */
:root {
    color-scheme: light dark;
    --background-color: light-dark(#ffffff, #0c0f0f);
    --text-color: light-dark(#1f2937, #f8fafc);
    --home-color: light-dark(#0284c7, #38bdf8);
    --away-color: light-dark(#dc2626, #f87171);
    --draw-color: light-dark(#d97706, #e5c158);
    --text-muted: light-dark(#4b5563, #8b9e9b);
    --title-color: light-dark(#0f172a, #f8fafc);
    --card-bg: color-mix(in srgb, var(--background-color), var(--text-color) 4%);
    --card-border: color-mix(in srgb, var(--background-color), var(--text-color) 12%);
    --card-bg-hover: color-mix(in srgb, var(--background-color), var(--text-color) 8%);
    --btn-bg: color-mix(in srgb, var(--background-color), var(--text-color) 6%);
    --btn-hover: color-mix(in srgb, var(--background-color), var(--text-color) 10%);
}

/* Global CSS Overrides */
html, body, [class*="css"] { 
    font-family: 'Plus Jakarta Sans', sans-serif; 
}

.stApp {
    background-color: var(--background-color) !important;
    color: var(--text-color) !important;
}

[data-testid="stSidebar"] {
    background-color: color-mix(in srgb, var(--background-color), var(--text-color) 3%) !important;
    border-right: 1px solid var(--card-border) !important;
}

/* Style selectboxes and inputs in sidebar */
div[data-baseweb="select"] {
    cursor: pointer !important;
}
div[data-baseweb="select"] > div {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    color: var(--text-color) !important;
    border-radius: 6px !important;
    cursor: pointer !important;
}
div[data-testid="stSelectbox"] div, 
div[data-testid="stSelectbox"] span, 
div[data-testid="stSelectbox"] svg {
    cursor: pointer !important;
}
div[role="listbox"] {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
}
div[role="option"], 
div[role="option"] * {
    color: var(--text-color) !important;
    background-color: transparent !important;
    cursor: pointer !important;
}
div[role="option"]:hover {
    background-color: var(--card-bg-hover) !important;
}

/* Expander containers */
div[data-testid="stExpander"] {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}
div[data-testid="stExpander"] details summary p {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    color: var(--title-color) !important;
    text-transform: uppercase !important;
}

/* Custom button overrides */
div.stButton > button, div.stDownloadButton > button {
    background: var(--btn-bg) !important;
    color: var(--draw-color) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important;
    height: 2.5rem;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    background: var(--btn-hover) !important;
    border-color: #10b981 !important;
    color: #10b981 !important;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.2) !important;
}

/* Sidebar Headings & Labels */
.sidebar-header {
    font-family: 'Outfit', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    margin-top: 1rem;
    margin-bottom: 1.25rem;
    text-transform: uppercase;
    border-bottom: 1px solid var(--card-border);
    padding-bottom: 0.25rem;
}
div[data-testid="stSidebar"] label p {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    margin-bottom: 0.2rem !important;
}

/* Tactical Header */
.tactical-header {
    text-align: center;
    padding: 1.5rem 0 1rem 0;
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--card-border);
}
.telemetry-badge {
    display: inline-block;
    background: rgba(16, 185, 129, 0.1);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 0.25rem 0.75rem;
    font-size: 0.75rem;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    letter-spacing: 0.12em;
    border-radius: 4px;
    margin-bottom: 0.75rem;
}
.tactical-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--title-color);
    letter-spacing: -0.02em;
    margin: 0;
    display: flex;
    align-items: center;
    justify-content: center;
}
.tactical-subtitle {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-top: 0.5rem;
    margin-bottom: 0;
}

/* Plotly Custom Theme-Adaptability Overrides */
.js-line, .gridlayer path {
    stroke: var(--card-border) !important;
}
.xtick text, .ytick text {
    fill: var(--text-muted) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.gtitle {
    fill: var(--title-color) !important;
    font-family: 'Outfit', sans-serif !important;
}
.bartext, .bartext text {
    fill: var(--title-color) !important;
    color: var(--title-color) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Slider overrides */
div[data-testid="stSlider"] label p {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
}

/* Small screen layout adjustments */
@media (max-width: 768px) {
    .matchup-billboard {
        flex-direction: column;
        padding: 1.5rem;
        text-align: center;
    }
    .team-panel {
        flex-direction: column;
        align-items: center;
        gap: 0.75rem;
    }
    .away-panel {
        flex-direction: column;
        text-align: center;
    }
    .vs-divider {
        border-left: none;
        border-right: none;
        border-top: 1px solid var(--card-border);
        border-bottom: 1px solid var(--card-border);
        padding: 1rem 0;
        width: 100%;
    }
}
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# Country → ISO-2 Mapper for FlagCDN
# ══════════════════════════════════════════════════════════════════════════════

COUNTRY_TO_ISO = {
    "england": "gb-eng", "france": "fr", "germany": "de", "italy": "it", "spain": "es",
    "portugal": "pt", "belgium": "be", "netherlands": "nl", "croatia": "hr", "denmark": "dk",
    "switzerland": "ch", "poland": "pl", "ukraine": "ua", "scotland": "gb-sct", "wales": "gb-wls",
    "sweden": "se", "austria": "at", "turkey": "tr", "hungary": "hu", "czechia": "cz",
    "czech republic": "cz", "slovakia": "sk", "romania": "ro", "serbia": "rs", "slovenia": "si",
    "norway": "no", "finland": "fi", "greece": "gr", "republic of ireland": "ie", "ireland": "ie",
    "northern ireland": "gb-nir", "albania": "al", "georgia": "ge", "iceland": "is",
    "north macedonia": "mk", "bosnia and herzegovina": "ba", "bulgaria": "bg", "montenegro": "me",
    "belarus": "by", "lithuania": "lt", "latvia": "lv", "estonia": "ee", "luxembourg": "lu",
    "cyprus": "cy", "malta": "mt", "andorra": "ad", "san marino": "sm", "liechtenstein": "li",
    "gibraltar": "gi", "faroe islands": "fo", "kosovo": "xk", "israel": "il", "azerbaijan": "az",
    "armenia": "am", "kazakhstan": "kz",
    "argentina": "ar", "brazil": "br", "uruguay": "uy", "colombia": "co", "peru": "pe",
    "chile": "cl", "paraguay": "py", "ecuador": "ec", "venezuela": "ve", "bolivia": "bo",
    "usa": "us", "united states": "us", "mexico": "mx", "canada": "ca", "costa rica": "cr",
    "jamaica": "jm", "panama": "pa", "honduras": "hn", "el salvador": "sv", "haiti": "ht",
    "curacao": "cw", "trinidad and tobago": "tt", "martinique": "mq", "guadeloupe": "gp",
    "guatemala": "gt", "cuba": "cu", "suriname": "sr", "french guiana": "gf", "bermuda": "bm",
    "grenada": "gd", "barbados": "bb", "dominica": "dm", "saint lucia": "lc",
    "saint vincent and the grenadines": "vc", "saint kitts and nevis": "kn", "antigua and barbuda": "ag",
    "montserrat": "ms", "virgin islands": "vi", "puerto rico": "pr", "cayman islands": "ky",
    "bahamas": "bs", "belize": "bz", "nicaragua": "ni", "guyana": "gy",
    "senegal": "sn", "morocco": "ma", "nigeria": "ng", "egypt": "eg", "tunisia": "tn",
    "cameroon": "cm", "algeria": "dz", "ghana": "gh", "ivory coast": "ci", "cote d'ivoire": "ci",
    "mali": "ml", "burkina faso": "bf", "dr congo": "cd", "congo dr": "cd", "south africa": "za",
    "cape verde": "cv", "cabo verde": "cv", "guinea": "gn", "equatorial guinea": "gq", "gabon": "ga",
    "togo": "tg", "angola": "ao", "zambia": "zm", "uganda": "ug", "kenya": "ke", "zimbabwe": "zw",
    "tanzania": "tz", "libya": "ly", "madagascar": "mg", "mauritania": "mr", "namibia": "na",
    "guinea-bissau": "gw", "mozambique": "mz", "malawi": "mw", "sudan": "sd", "south sudan": "ss",
    "ethiopia": "et", "rwanda": "rw", "burundi": "bi", "central african republic": "cf",
    "congo": "cg", "republic of the congo": "cg", "chad": "td", "niger": "ne", "benin": "bj",
    "sierra leone": "sl", "liberia": "lr", "gambia": "gm", "eritrea": "er", "somalia": "so",
    "djibouti": "dj", "seychelles": "sc", "mauritius": "mu", "comoros": "km", "reunion": "re",
    "japan": "jp", "iran": "ir", "ir iran": "ir", "south korea": "kr", "korea republic": "kr",
    "australia": "au", "saudi arabia": "sa", "qatar": "qa", "iraq": "iq", "uae": "ae",
    "united arab emirates": "ae", "oman": "om", "uzbekistan": "uz", "china": "cn", "china pr": "cn",
    "jordan": "jo", "bahrain": "bh", "syria": "sy", "palestine": "ps", "vietnam": "vn",
    "kyrgyzstan": "kg", "lebannon": "lb", "lebanon": "lb", "india": "in", "tajikistan": "tj",
    "thailand": "th", "north korea": "kp", "korea dpr": "kp", "yemen": "ye", "kuwait": "kw",
    "afghanistan": "af", "turkmenistan": "tm", "maldives": "mv", "nepal": "np", "bangladesh": "bd",
    "pakistan": "pk", "sri lanka": "lk", "bhutan": "bt", "guam": "gu", "macau": "mo",
    "chinese taipei": "tw", "taiwan": "tw", "hong kong": "hk", "myanmar": "mm", "cambodia": "kh",
    "laos": "la", "brunei": "bn", "timor-leste": "tl", "singapore": "sg", "malaysia": "my",
    "indonesia": "id", "philippines": "ph",
    "new zealand": "nz", "solomon islands": "sb", "new caledonia": "nc", "tahiti": "pf",
    "fiji": "fj", "vanuatu": "vu", "papua new guinea": "pg", "samoa": "ws", "american samoa": "as",
    "tonga": "to", "cook islands": "ck",
}


def get_flag_url(team_name: str) -> str:
    """Return FlagCDN URL for a team's flag."""
    name_clean = team_name.lower().strip()
    iso = COUNTRY_TO_ISO.get(name_clean)
    if not iso:
        return "https://flagcdn.com/w80/un.png"  # UN flag default
    return f"https://flagcdn.com/w80/{iso}.png"


def inject_theme():
    """Inject the shared CSS theme into the current Streamlit page."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)

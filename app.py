"""
InternAI – Smart Internship Trust & Skill-Fit Analyzer
app.py  –  Streamlit application

Compatible with the artifact saved by train_model.py.
Run:
    streamlit run app.py
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import joblib
import plotly.graph_objects as go
import streamlit as st

# ── Path constants ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "internai_model.pkl")

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InternAI – Internship Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Dark navy dashboard styling ────────────────────────────────────────────────
STYLE = """
<style>
  /* Global */
  html, body, [data-testid="stAppViewContainer"] {
      background-color: #0d1b2a;
      color: #e8eaf0;
      font-family: 'Segoe UI', sans-serif;
  }
  [data-testid="stHeader"] { background-color: #0d1b2a; }
  [data-testid="stSidebar"] { background-color: #111f30; }

  /* Headings */
  h1 { color: #4fc3f7 !important; }
  h2, h3, h4 { color: #81d4fa !important; }

  /* Cards / sections */
  .card {
      background: #112233;
      border: 1px solid #1e3a5f;
      border-radius: 10px;
      padding: 1.1rem 1.4rem;
      margin-bottom: 0.9rem;
  }
  .card-warn  { border-left: 4px solid #ff7043; }
  .card-ok    { border-left: 4px solid #66bb6a; }
  .card-info  { border-left: 4px solid #4fc3f7; }
  .card-caution { border-left: 4px solid #ffa726; }

  /* Badges */
  .badge {
      display: inline-block;
      padding: 0.18rem 0.65rem;
      border-radius: 20px;
      font-size: 0.78rem;
      font-weight: 600;
      margin: 0.15rem;
  }
  .badge-red    { background: #b71c1c; color: #ffcdd2; }
  .badge-orange { background: #e65100; color: #ffe0b2; }
  .badge-green  { background: #1b5e20; color: #c8e6c9; }
  .badge-blue   { background: #0d47a1; color: #bbdefb; }
  .badge-grey   { background: #37474f; color: #cfd8dc; }

  /* Text utilities */
  .muted  { color: #8fa3b8; font-size: 0.85rem; }
  .mono   { font-family: 'Courier New', monospace; font-size: 0.82rem; }

  /* Metric strip */
  .metric-row {
      display: flex; flex-wrap: wrap; gap: 0.8rem; margin: 0.5rem 0;
  }
  .metric-box {
      background: #0d2137; border: 1px solid #1e3a5f; border-radius: 8px;
      padding: 0.6rem 1rem; text-align: center; min-width: 110px;
  }
  .metric-val { font-size: 1.6rem; font-weight: 700; color: #4fc3f7; }
  .metric-lbl { font-size: 0.75rem; color: #8fa3b8; }

  /* Streamlit widget tweaks */
  .stTextArea textarea, .stTextInput input {
      background: #0d2137 !important;
      color: #e8eaf0 !important;
      border: 1px solid #1e3a5f !important;
  }
  .stButton > button {
      background: linear-gradient(135deg, #1565c0, #0288d1);
      color: white; border: none; border-radius: 8px;
      padding: 0.55rem 1.6rem; font-size: 1rem; font-weight: 600;
  }
  .stButton > button:hover { opacity: 0.9; }
  hr { border-color: #1e3a5f; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODEL LOADING                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@st.cache_resource(show_spinner="Loading InternAI model …")
def load_model_artifact() -> Dict[str, Any]:
    """Load the joblib artifact produced by train_model.py."""
    if not os.path.exists(MODEL_PATH):
        return {}
    return joblib.load(MODEL_PATH)


# Sentinel value returned as raw_label when max-class probability is too low
# to make a meaningful prediction.  app.py checks for this string explicitly.
INCONCLUSIVE_LABEL = "__inconclusive__"


def predict_label(
    artifact: Dict[str, Any], description: str
) -> Tuple[str, float, Dict[str, float]]:
    """
    Run the ML pipeline on the description text.

    Returns (raw_label, max_proba, all_probas) where:
      raw_label  — original class string or INCONCLUSIVE_LABEL
      max_proba  — highest class probability
      all_probas — dict {class_name: probability} for all classes

    The threshold is read from the artifact (default 0.55).
    """
    pipeline      = artifact["pipeline"]
    label_encoder = artifact.get("label_encoder")
    class_names   = artifact.get("class_names", [])
    threshold     = float(artifact.get("inconclusive_threshold", 0.55))

    text_input = [str(description).strip()]

    # Always use predict_proba so we have probabilities before deciding
    if not hasattr(pipeline, "predict_proba"):
        # Fallback: no probability support
        numeric_pred = pipeline.predict(text_input)[0]
        if label_encoder is not None:
            raw_label = str(label_encoder.inverse_transform([int(numeric_pred)])[0])
        elif class_names:
            raw_label = class_names[int(numeric_pred)]
        else:
            raw_label = str(numeric_pred)
        return raw_label, 0.0, {}

    proba     = pipeline.predict_proba(text_input)[0]
    max_proba = float(max(proba))
    pred_idx  = int(proba.argmax())

    # Build full probability dict aligned to class_names
    all_probas = {cls: float(p) for cls, p in zip(class_names, proba)}

    # Decode the predicted class index to its original string label
    if label_encoder is not None:
        raw_label = str(label_encoder.inverse_transform([pred_idx])[0])
    elif class_names:
        raw_label = class_names[pred_idx]
    else:
        raw_label = str(pred_idx)

    # If confidence is too low, report as inconclusive rather than misleading
    if max_proba < threshold:
        return INCONCLUSIVE_LABEL, max_proba, all_probas

    return raw_label, max_proba, all_probas


def map_label_to_dashboard(raw_label: str) -> Dict[str, str]:
    """Map any raw class name to dashboard display properties."""
    if raw_label == INCONCLUSIVE_LABEL:
        return {
            "display": "Inconclusive – manual verification needed",
            "badge"  : "badge-grey",
            "card"   : "card-info",
        }
    label_lower = raw_label.lower().strip()
    if "fraud" in label_lower:
        return {"display": "Possibly Fraudulent",  "badge": "badge-red",    "card": "card-warn"}
    if "suspicious" in label_lower or "suspect" in label_lower:
        return {"display": "Suspicious",           "badge": "badge-orange", "card": "card-caution"}
    if "legit" in label_lower or "genuine" in label_lower or "safe" in label_lower:
        return {"display": "Likely Legitimate",    "badge": "badge-green",  "card": "card-ok"}
    # Unknown class – show as-is
    return {"display": raw_label.title(),          "badge": "badge-blue",   "card": "card-info"}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RULE-BASED WARNING ANALYSIS                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Pattern groups: (label, regex_pattern)
WARNING_PATTERNS: List[Tuple[str, str]] = [
    ("Upfront payment request",
     r"\b(pay|payment|fee|charge|deposit|cost|invest)\b.{0,60}\b(register|registration|secure|apply|application|join|start|training|kit)\b"),
    ("Registration or application fee",
     r"\b(registration|application|processing|admin(?:istration)?)\s+(fee|charge|cost|payment)\b"),
    ("Unrealistic salary promise",
     r"\b(earn|make|income|salary|pay).{0,40}\b(\d[\d,]*\s*(k|thousand|lakh|per\s*(day|week|hour)))\b"),
    ("Urgent payment pressure",
     r"\b(urgent(?:ly)?|immediately|act\s+now|limited\s+spots?|hurry|don[''']t\s+miss|last\s+chance)\b"),
    ("Request for personal financial details",
     r"\b(bank\s+account|credit\s+card|debit\s+card|wire\s+transfer|western\s+union|paypal|upi\s+id|ifsc)\b"),
    ("Work-from-home with no experience needed",
     r"\b(work\s+from\s+home|remote|online\s+job).{0,80}no\s+(experience|qualification|skill)\b"),
    ("Too-good salary with no requirements",
     r"\b(high\s+salary|huge\s+pay|big\s+income).{0,80}no\s+(experience|degree|qualification)\b"),
    ("Vague company details",
     r"\b(confidential\s+company|company\s+name\s+withheld|undisclosed\s+company)\b"),
    ("Requests for ID or documents upfront",
     r"\b(send|submit|upload|share|provide).{0,40}\b(passport|aadhar|pan\s+card|national\s+id|ssn|driving\s+licence)\b"),
    ("Promise of guaranteed job",
     r"\b(guaranteed?\s+(job|placement|hiring|selection)|100\s*%\s*(placement|hired))\b"),
]

# Negation prefixes — if any of these immediately precede the pattern match, skip it
NEGATION_RE = re.compile(
    r"\b(no|not|never|without|free|waive[ds]?|don[''']t|doesn[''']t|won[''']t|exempt|zero)\b"
    r"[\s\w,]{0,30}$",
    re.IGNORECASE,
)


def _is_negated(text: str, match_start: int) -> bool:
    """Check if the 40-character window before the match contains a negation phrase."""
    window = text[max(0, match_start - 120): match_start]
    return bool(NEGATION_RE.search(window))


def analyze_warnings(description: str) -> Dict[str, Any]:
    """
    Scan description for red-flag patterns while respecting negations.
    Returns a dict with:
        'warnings': list of dicts {"label": str, "snippet": str}
        'count'   : int
    """
    text   = description.lower()
    found  = []
    seen   = set()

    for label, pattern in WARNING_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            if label in seen:
                break
            if _is_negated(text, m.start()):
                continue
            # Extract a readable snippet from the original (mixed-case) text
            start   = max(0, m.start() - 20)
            end     = min(len(description), m.end() + 40)
            snippet = "…" + description[start:end].strip() + "…"
            found.append({"label": label, "snippet": snippet})
            seen.add(label)
            break

    return {"warnings": found, "count": len(found)}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SKILL MATCHING                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Alias map: alternative → canonical form
SKILL_ALIASES: Dict[str, str] = {
    "ml"                       : "machine learning",
    "ai"                       : "artificial intelligence",
    "dl"                       : "deep learning",
    "py"                       : "python",
    "js"                       : "javascript",
    "ts"                       : "typescript",
    "react.js"                 : "react",
    "reactjs"                  : "react",
    "node.js"                  : "node",
    "nodejs"                   : "node",
    "vue.js"                   : "vue",
    "vuejs"                    : "vue",
    "express.js"               : "express",
    "expressjs"                : "express",
    "mongo"                    : "mongodb",
    "postgres"                 : "postgresql",
    "k8s"                      : "kubernetes",
    "tf"                       : "tensorflow",
    "sklearn"                  : "scikit-learn",
    "scikit learn"             : "scikit-learn",
    "nlp"                      : "natural language processing",
    "cv"                       : "computer vision",
    "oop"                      : "object-oriented programming",
    "rest"                     : "rest api",
    "restful"                  : "rest api",
    "sql server"               : "mssql",
    "microsoft sql server"     : "mssql",
    "c++"                      : "cpp",
    "c#"                       : "csharp",
    "html5"                    : "html",
    "css3"                     : "css",
    "scss"                     : "css",
    "sass"                     : "css",
    "aws"                      : "amazon web services",
    "gcp"                      : "google cloud platform",
    "ms excel"                 : "excel",
    "microsoft excel"          : "excel",
    "ms word"                  : "word",
    "microsoft word"           : "word",
    "ms office"                : "microsoft office",
    "office suite"             : "microsoft office",
}

# Skills the system can suggest when a required skill is missing
LEARNING_RESOURCES: Dict[str, str] = {
    "python"               : "Python – python.org/doc",
    "javascript"           : "JavaScript – javascript.info",
    "machine learning"     : "ML – fast.ai / Coursera",
    "deep learning"        : "DL – fast.ai / deeplearning.ai",
    "data analysis"        : "Data Analysis – Kaggle Learn",
    "sql"                  : "SQL – sqlzoo.net",
    "react"                : "React – react.dev",
    "node"                 : "Node.js – nodejs.org/learn",
    "docker"               : "Docker – docs.docker.com/get-started",
    "kubernetes"           : "Kubernetes – kubernetes.io/docs",
    "amazon web services"  : "AWS – aws.amazon.com/training",
    "google cloud platform": "GCP – cloud.google.com/learn",
    "scikit-learn"         : "scikit-learn – scikit-learn.org/stable",
    "tensorflow"           : "TensorFlow – tensorflow.org/learn",
    "pytorch"              : "PyTorch – pytorch.org/tutorials",
    "natural language processing": "NLP – huggingface.co/course",
    "git"                  : "Git – git-scm.com/doc",
    "linux"                : "Linux – linuxcommand.org",
    "excel"                : "Excel – support.microsoft.com",
    "communication"        : "Communication – coursera.org",
    "teamwork"             : "Teamwork – LinkedIn Learning",
    "rest api"             : "REST APIs – restfulapi.net",
    "mongodb"              : "MongoDB – mongodb.com/docs",
    "postgresql"           : "PostgreSQL – postgresql.org/docs",
}


def _parse_skills(raw: str) -> List[str]:
    """Split and normalise a comma/semicolon/newline separated skill string."""
    if not raw or not raw.strip():
        return []
    tokens = re.split(r"[,;\n]+", raw)
    normalised = []
    for tok in tokens:
        cleaned = tok.strip().lower()
        if cleaned:
            # Resolve alias
            canonical = SKILL_ALIASES.get(cleaned, cleaned)
            normalised.append(canonical)
    return list(dict.fromkeys(normalised))  # deduplicate, preserve order


def match_skills(
    student_skills_raw: str, required_skills_raw: str
) -> Dict[str, Any]:
    """
    Compare student skills against required skills.
    Returns dict with keys: matched, missing, percentage, suggestions, total_required.
    """
    student  = set(_parse_skills(student_skills_raw))
    required = _parse_skills(required_skills_raw)

    if not required:
        return {
            "matched"        : [],
            "missing"        : [],
            "percentage"     : 0.0,
            "suggestions"    : [],
            "total_required" : 0,
        }

    required_set = set(required)
    matched  = sorted(student & required_set)
    missing  = sorted(required_set - student)
    pct      = 100.0 * len(matched) / len(required_set)

    # Suggest learning resources for missing skills
    suggestions = []
    for skill in missing:
        resource = LEARNING_RESOURCES.get(skill)
        if resource:
            suggestions.append(resource)
        else:
            # Generic suggestion
            suggestions.append(f"{skill.title()} – search on Coursera / YouTube")

    return {
        "matched"        : matched,
        "missing"        : missing,
        "percentage"     : round(pct, 1),
        "suggestions"    : suggestions,
        "total_required" : len(required_set),
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  UI COMPONENTS                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def render_header() -> None:
    st.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 0.8rem;">
      <h1 style="font-size:2.2rem; letter-spacing:1px;">🔍 InternAI</h1>
      <p style="color:#8fa3b8; font-size:1.05rem; margin-top:-0.5rem;">
        Smart Internship Trust &amp; Skill-Fit Analyzer
      </p>
    </div>
    <hr>
    """, unsafe_allow_html=True)


def render_ml_result(
    raw_label: str, confidence: float, artifact: Dict[str, Any], warning_count: int = 0,
    all_probas: Optional[Dict[str, float]] = None
) -> None:
    """Render the ML classification card.

    all_probas: optional dict mapping class name -> probability (0-1), used to
    display the full per-class breakdown so the user can see how close the
    three classes are.
    """
    display_map = map_label_to_dashboard(raw_label)
    card_cls    = display_map["card"]
    badge_cls   = display_map["badge"]
    display_lbl = display_map["display"]
    threshold   = float(artifact.get("inconclusive_threshold", 0.55))
    class_names = artifact.get("class_names", [])

    # Training accuracy from artifact metrics
    train_acc = artifact.get("metrics", {}).get("accuracy")
    acc_note  = f" | Model accuracy: {train_acc*100:.1f}%" if train_acc else ""

    is_inconclusive = (raw_label == INCONCLUSIVE_LABEL)
    label_lower     = raw_label.lower()

    # ── Per-class probability bar ──────────────────────────────────────────────
    # Shown whenever we have the full probability vector.  Lets the user see how
    # close the three classes are, which is the key information when confidence
    # is marginal.
    proba_bar_html = ""
    if all_probas:
        rows = ""
        bar_colours = {
            "fraudulent": "#ef5350",
            "legitimate": "#66bb6a",
            "suspicious": "#ffa726",
        }
        for cls in class_names:
            p   = all_probas.get(cls, 0.0)
            pct = round(p * 100, 1)
            col = bar_colours.get(cls, "#4fc3f7")
            rows += (
                f'<tr>'
                f'<td style="color:#8fa3b8;width:90px;font-size:0.8rem;">{cls}</td>'
                f'<td style="width:140px;">'
                f'  <div style="background:#1e3a5f;border-radius:3px;height:8px;">'
                f'    <div style="background:{col};width:{pct}%;height:100%;border-radius:3px;"></div>'
                f'  </div>'
                f'</td>'
                f'<td style="color:#8fa3b8;font-size:0.8rem;padding-left:6px;">{pct}%</td>'
                f'</tr>'
            )
        proba_bar_html = (
            f'<table style="border-collapse:collapse;margin:0.5rem 0 0.2rem;">'
            f'{rows}</table>'
        )

    # ── False-positive note ────────────────────────────────────────────────────
    # Shown when the model predicts suspicious/fraudulent but no rule-based
    # warning phrases fired.  The specific signal context (entry-level sales,
    # business development) comes from corpus analysis of the training data.
    is_flagged_no_warnings = (
        not is_inconclusive
        and warning_count == 0
        and ("suspicious" in label_lower or "suspect" in label_lower or "fraud" in label_lower)
    )
    false_positive_note = (
        '<p class="muted" style="margin-top:0.4rem; border-top:1px solid #1e3a5f; padding-top:0.4rem;">'
        '<strong>Note:</strong> The ML model flagged this posting, but no configured warning '
        'phrases were detected. This is likely a false positive — phrases such as '
        '"entry level sales", "business development", "no experience required", '
        '"work from home", and "no screening questions" are disproportionately '
        'associated with the suspicious label in the training data, even when the '
        'posting itself contains no scam signals. '
        'Please verify the posting independently before drawing any conclusions.'
        '</p>'
        if is_flagged_no_warnings else ""
    )

    if is_inconclusive:
        conf_html = (
            f'<p class="muted" style="margin-top:0.5rem;">'
            f'Highest class probability: <strong>{confidence*100:.1f}%</strong>'
            f' — below the {threshold*100:.0f}% threshold required for a confident prediction.'
            f'</p>'
            f'<p class="muted">'
            f'The model cannot reliably separate the three classes for this text. '
            f'This often happens when the description does not contain strong signals '
            f'in either direction, or when it uses phrases common to both legitimate '
            f'and suspicious postings (e.g. "entry level", "work from home"). '
            f'Inspect the rule-based warnings below and verify the posting independently.'
            f'</p>'
        )
        label_note = "Inconclusive"
    else:
        conf_html = (
            f'<p class="muted" style="margin-top:0.5rem;">'
            f'Model confidence: <strong>{confidence*100:.1f}%</strong>'
            f' — the model\'s internal score, <em>not</em> a verified fraud probability.'
            f'</p>'
        )
        label_note = f'raw label: <span class="mono">{raw_label}</span>'

    st.markdown(f"""
    <div class="card {card_cls}">
      <span class="badge {badge_cls}">{display_lbl}</span>
      {conf_html}
      {proba_bar_html}
      {false_positive_note}
      <p class="muted">ML classification ({label_note}){acc_note}</p>
      <p class="muted"><em>This is an automated analysis only. Always verify internships independently.</em></p>
    </div>
    """, unsafe_allow_html=True)


def render_warnings(warning_result: Dict[str, Any]) -> None:
    """Render the rule-based warning cards."""
    warnings = warning_result["warnings"]
    count    = warning_result["count"]

    if count == 0:
        st.markdown("""
        <div class="card card-ok">
          <strong>✅ No automated red flags detected</strong>
          <p class="muted" style="margin:0.3rem 0 0;">
            No common scam patterns were found. This does not guarantee legitimacy.
          </p>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown(f"""
    <div class="card card-warn">
      <strong>⚠️ {count} warning indicator{"s" if count > 1 else ""} detected</strong>
    </div>
    """, unsafe_allow_html=True)

    for w in warnings:
        st.markdown(f"""
        <div class="card card-caution" style="margin-left:1rem;">
          <span class="badge badge-orange">🔶 {w['label']}</span>
          <p class="mono" style="margin:0.4rem 0 0; color:#cfd8dc;">
            {w['snippet']}
          </p>
        </div>
        """, unsafe_allow_html=True)


def render_skill_chart(skill_result: Dict[str, Any]) -> None:
    """Render a Plotly bar chart of matched vs missing skills."""
    matched = skill_result["matched"]
    missing = skill_result["missing"]

    if not matched and not missing:
        return

    labels   = matched + missing
    values   = [1] * len(matched) + [1] * len(missing)
    colours  = ["#66bb6a"] * len(matched) + ["#ef5350"] * len(missing)
    statuses = ["Matched"] * len(matched) + ["Missing"] * len(missing)

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colours,
        text=statuses,
        textposition="inside",
        hovertemplate="<b>%{x}</b><br>Status: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title="Skills Overview",
        xaxis_title="Skill",
        yaxis_visible=False,
        plot_bgcolor="#0d1b2a",
        paper_bgcolor="#112233",
        font_color="#e8eaf0",
        title_font_color="#4fc3f7",
        xaxis=dict(tickfont=dict(color="#e8eaf0"), gridcolor="#1e3a5f"),
        margin=dict(t=50, b=10, l=10, r=10),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_skill_results(skill_result: Dict[str, Any]) -> None:
    """Render skill matching section."""
    pct           = skill_result["percentage"]
    matched       = skill_result["matched"]
    missing       = skill_result["missing"]
    suggestions   = skill_result["suggestions"]
    total_req     = skill_result["total_required"]

    if total_req == 0:
        st.markdown("""
        <div class="card card-info">
          <p class="muted">Enter required skills to see your skill-fit analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Colour threshold
    if pct >= 75:
        bar_colour, badge_cls = "#66bb6a", "badge-green"
    elif pct >= 40:
        bar_colour, badge_cls = "#ffa726", "badge-orange"
    else:
        bar_colour, badge_cls = "#ef5350", "badge-red"

    matched_html  = "".join(f'<span class="badge badge-green">{s}</span>' for s in matched) or \
                    '<span class="muted">None matched</span>'
    missing_html  = "".join(f'<span class="badge badge-red">{s}</span>' for s in missing) or \
                    '<span class="muted">None — you have all required skills!</span>'

    st.markdown(f"""
    <div class="card card-info">
      <h3 style="margin-top:0;">📊 Skill-Fit Analysis</h3>
      <div class="metric-row">
        <div class="metric-box">
          <div class="metric-val">{pct:.0f}%</div>
          <div class="metric-lbl">Skill Match</div>
        </div>
        <div class="metric-box">
          <div class="metric-val">{len(matched)}/{total_req}</div>
          <div class="metric-lbl">Skills Matched</div>
        </div>
      </div>
      <div style="background:#1e3a5f; border-radius:6px; height:10px; margin:0.7rem 0;">
        <div style="background:{bar_colour}; width:{pct}%; height:100%; border-radius:6px;"></div>
      </div>
      <p><strong>✅ Matched Skills:</strong><br>{matched_html}</p>
      <p><strong>❌ Missing Skills:</strong><br>{missing_html}</p>
    </div>
    """, unsafe_allow_html=True)

    if suggestions:
        sugg_items = "".join(f"<li style='margin:0.2rem 0;'>{s}</li>" for s in suggestions)
        st.markdown(f"""
        <div class="card card-info">
          <h4 style="margin-top:0;">📚 Suggested Skills to Learn</h4>
          <ul style="padding-left:1.2rem; color:#8fa3b8;">{sugg_items}</ul>
        </div>
        """, unsafe_allow_html=True)

    render_skill_chart(skill_result)


def render_next_steps(warning_count: int, raw_label: str, pct: float) -> None:
    """Render context-aware recommended next steps."""
    steps = []
    label_lower = raw_label.lower()
    is_inconclusive = (raw_label == INCONCLUSIVE_LABEL)

    if warning_count >= 3 or "fraud" in label_lower:
        steps += [
            "Do not share personal documents or pay any fee.",
            "Search for the company's official website independently.",
            "Report suspicious postings to the relevant job-portal trust & safety team.",
        ]
    elif is_inconclusive or "suspicious" in label_lower or warning_count >= 1:
        steps += [
            "The model could not make a confident classification — treat this posting with caution.",
            "Research the company on LinkedIn or Glassdoor before proceeding.",
            "Ask for a formal offer letter and verify the company's contact details.",
            "Speak to current/former interns if possible.",
        ]
    else:
        steps += [
            "The posting appears relatively safe — still do your own due diligence.",
            "Prepare your CV and a tailored cover letter.",
            "Reach out to the company's HR via the official contact listed.",
        ]

    if pct < 60:
        steps.append("📚 Upskill in the missing required areas before applying.")
    elif pct >= 90:
        steps.append("🎯 Excellent skill fit — highlight your top matching skills in your application.")

    items_html = "".join(f"<li style='margin:0.3rem 0;'>{s}</li>" for s in steps)
    st.markdown(f"""
    <div class="card card-info">
      <h3 style="margin-top:0;">🗺️ Recommended Next Steps</h3>
      <ul style="padding-left:1.2rem;">{items_html}</ul>
    </div>
    """, unsafe_allow_html=True)


def render_limitations() -> None:
    """Render limitations and responsible-use section."""
    st.markdown("""
    <div class="card" style="border-left: 4px solid #546e7a; margin-top:1.5rem;">
      <h4 style="margin-top:0; color:#90a4ae;">⚖️ Limitations &amp; Responsible Use</h4>
      <ul style="color:#8fa3b8; font-size:0.88rem; padding-left:1.2rem;">
        <li>InternAI uses machine learning and rule-based heuristics — it <strong>cannot</strong> definitively confirm fraud.</li>
        <li>Model confidence is an internal score, <strong>not</strong> a verified probability of fraud or legitimacy.</li>
        <li><strong>Suspicious class limitation:</strong> the model correctly identifies only ~68% of suspicious postings.
            Many suspicious internships are flagged by structural signals (missing company logo, missing requirements)
            that are not detectable from text alone. A "Suspicious" prediction with no rule-based warnings may be a
            false positive; a "Legitimate" or "Inconclusive" result does not rule out structural red flags.</li>
        <li>Legitimate postings may trigger warnings; fraudulent ones may evade detection.</li>
        <li>Always cross-verify using official company websites, LinkedIn, and trusted job portals.</li>
        <li>Do not make irreversible decisions (paying fees, sharing documents) based solely on this tool.</li>
        <li>The model was trained on a 900-sample dataset and may not generalise to all regions or industries.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN APP                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    render_header()

    # ── Load model artifact ────────────────────────────────────────────────────
    artifact = load_model_artifact()

    if not artifact:
        st.error(
            "⚠️ **Model file not found.**\n\n"
            "Run the training script first:\n\n"
            "```bash\n"
            "python train_model.py\n"
            "```\n\n"
            "Then refresh this page."
        )
        st.stop()

    # ── Sidebar: model info ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🤖 Model Info")
        class_names = artifact.get("class_names", [])
        st.markdown(f"**Classes:** {', '.join(class_names) if class_names else 'N/A'}")
        metrics = artifact.get("metrics", {})
        if metrics:
            st.markdown(f"**Accuracy:** {metrics.get('accuracy', 0)*100:.1f}%")
            st.markdown(f"**F1 (weighted):** {metrics.get('f1', 0)*100:.1f}%")
            if metrics.get("f1_macro"):
                st.markdown(f"**F1 (macro):** {metrics.get('f1_macro', 0)*100:.1f}%")
        st.markdown("---")
        st.markdown(
            "<p class='muted'>InternAI uses ML + rule-based heuristics.<br>"
            "Results are indicative only.</p>",
            unsafe_allow_html=True,
        )

    # ── Input section ─────────────────────────────────────────────────────────
    st.markdown("## 📋 Internship Details")

    description = st.text_area(
        "Paste the internship description or job posting here",
        height=220,
        placeholder=(
            "e.g. We are hiring a Marketing Intern. You will assist with social media strategy,"
            " content creation, and analytics. No registration fee required."
        ),
    )

    col1, col2 = st.columns(2)
    with col1:
        student_skills_raw = st.text_area(
            "Your Skills",
            height=120,
            placeholder="e.g. Python, SQL, Excel, Communication",
        )
    with col2:
        required_skills_raw = st.text_area(
            "Required Skills (from the internship posting)",
            height=120,
            placeholder="e.g. Python, Machine Learning, SQL, Teamwork",
        )

    analyze_clicked = st.button("🔍 Analyze Internship", use_container_width=True)

    # ── Analysis ──────────────────────────────────────────────────────────────
    if analyze_clicked:
        if not description.strip():
            st.warning("Please paste an internship description to analyze.")
            st.stop()

        with st.spinner("Analyzing …"):
            # 1. ML classification
            raw_label, confidence, all_probas = predict_label(artifact, description)

            # 2. Rule-based warnings
            warning_result = analyze_warnings(description)

            # 3. Skill matching
            skill_result = match_skills(student_skills_raw, required_skills_raw)

        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        # ── ML result ─────────────────────────────────────────────────────────
        st.markdown("### 🤖 ML Classification")
        render_ml_result(raw_label, confidence, artifact, warning_result["count"], all_probas)

        # ── Warning indicators ─────────────────────────────────────────────────
        st.markdown("### 🚨 Rule-Based Warning Indicators")
        render_warnings(warning_result)

        # ── Skill fit ─────────────────────────────────────────────────────────
        st.markdown("### 🎓 Skill-Fit Analysis")
        render_skill_results(skill_result)

        # ── Next steps ─────────────────────────────────────────────────────────
        render_next_steps(
            warning_result["count"],
            raw_label,
            skill_result["percentage"],
        )

        # ── Limitations ────────────────────────────────────────────────────────
        render_limitations()

    else:
        # Landing placeholder
        st.markdown("""
        <div class="card card-info" style="text-align:center; padding:2rem;">
          <p style="font-size:1.1rem; color:#8fa3b8;">
            Paste an internship description above and click <strong>Analyze Internship</strong>
            to get your trust &amp; skill-fit report.
          </p>
        </div>
        """, unsafe_allow_html=True)
        render_limitations()


if __name__ == "__main__":
    main()

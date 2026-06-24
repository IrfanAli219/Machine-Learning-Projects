import streamlit as st
import joblib
import numpy as np
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DASS-42 Mental Health Assessment",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #F7F9FC;
}

#MainMenu, footer, header {visibility: hidden;}

.hero-banner {
    background: linear-gradient(135deg, #4A90D9 0%, #6B5EA8 100%);
    border-radius: 16px;
    padding: 40px 36px;
    margin-bottom: 28px;
    color: white;
    text-align: center;
}
.hero-banner h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero-banner p {
    font-size: 1rem;
    opacity: 0.88;
    margin: 0;
    line-height: 1.6;
}

.disclaimer-box {
    background: #FFF8E1;
    border-left: 4px solid #F5A623;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 24px;
    font-size: 0.875rem;
    color: #7A5C00;
    line-height: 1.6;
}

.section-card {
    background: white;
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #E8EDF3;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 4px;
}
.section-subtitle {
    font-size: 0.82rem;
    color: #6B7280;
    margin-bottom: 22px;
}

.q-label {
    font-size: 0.9rem;
    font-weight: 500;
    color: #1F2937;
    margin-bottom: 6px;
    margin-top: 14px;
    line-height: 1.5;
}

.result-wrapper {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    margin: 18px 0;
}
.result-card {
    flex: 1;
    min-width: 160px;
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    color: white;
}
.result-card .cat {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    opacity: 0.85;
    margin-bottom: 8px;
}
.result-card .sev {
    font-size: 1.3rem;
    font-weight: 700;
}
.result-card .conf {
    font-size: 0.75rem;
    opacity: 0.8;
    margin-top: 4px;
}

.sev-normal    { background: linear-gradient(135deg,#34D399,#059669); }
.sev-mild      { background: linear-gradient(135deg,#FCD34D,#D97706); }
.sev-moderate  { background: linear-gradient(135deg,#FB923C,#EA580C); }
.sev-severe    { background: linear-gradient(135deg,#F87171,#DC2626); }
.sev-ext       { background: linear-gradient(135deg,#C084FC,#7C3AED); }

.advice-box {
    background: #F0F7FF;
    border-radius: 10px;
    padding: 16px 20px;
    font-size: 0.875rem;
    color: #1E3A5F;
    line-height: 1.7;
    margin-top: 10px;
}
.advice-box b { color: #1D4ED8; }

div.stButton > button {
    background: linear-gradient(135deg,#4A90D9,#6B5EA8);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 14px 40px;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
    margin-top: 10px;
}
div.stButton > button:hover { opacity: 0.88; }

div[data-testid="stRadio"] > label {
    font-size: 0.88rem !important;
    color: #374151 !important;
}


</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DASS-42 QUESTIONS (Top 8 per category)
# ─────────────────────────────────────────────

STRESS_QUESTIONS = {
    'Q11 (S)': "I found myself getting upset rather easily.",
    'Q1 (S)':  "I found it hard to wind down.",
    'Q27 (S)': "I was unable to relax.",
    'Q29 (S)': "I found myself getting upset quite easily.",
    'Q8 (S)':  "I tended to over-react to situations.",
    'Q39 (S)': "I found it difficult to relax.",
    'Q12 (S)': "I felt that I was using a lot of nervous energy.",
    'Q33 (S)': "I found it hard to calm down after something upset me.",
}

ANXIETY_QUESTIONS = {
    'Q28 (A)': "I was aware of dryness of my mouth.",
    'Q36 (A)': "I felt scared without any good reason.",
    'Q9 (A)':  "I was worried about situations in which I might panic.",
    'Q7 (A)':  "I experienced trembling (e.g., in the hands).",
    'Q20 (A)': "I felt I was close to panic.",
    'Q40 (A)': "I felt terrified.",
    'Q41 (A)': "I was worried about situations in which I might embarrass myself.",
    'Q4 (A)':  "I experienced breathing difficulty (e.g., fast breathing).",
}

DEPRESSION_QUESTIONS = {
    'Q13 (D)': "I felt sad and depressed.",
    'Q34 (D)': "I was unable to become enthusiastic about anything.",
    'Q16 (D)': "I felt I had lost interest in just about everything.",
    'Q17 (D)': "I felt I wasn't worth much as a person.",
    'Q21 (D)': "I felt that life was meaningless.",
    'Q10 (D)': "I felt that I had nothing to look forward to.",
    'Q26 (D)': "I felt down-hearted and blue.",
    'Q24 (D)': "I couldn't seem to experience any positive feeling at all.",
}

RATING_OPTIONS = [
    "0 — Did not apply to me at all",
    "1 — Applied to me to some degree",
    "2 — Applied to me a considerable amount",
    "3 — Applied to me very much or most of the time",
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def severity_css(label: str) -> str:
    l = label.lower()
    if "normal" in l:     return "sev-normal"
    if "mild" in l:       return "sev-mild"
    if "moderate" in l:   return "sev-moderate"
    if "extremely" in l:  return "sev-ext"
    if "severe" in l:     return "sev-severe"
    return "sev-mild"

def get_advice(s_sev, a_sev, d_sev):
    levels = [s_sev, a_sev, d_sev]
    if all("normal" in x.lower() for x in levels):
        return ("<b>You're doing well!</b> All three dimensions fall within the normal range. "
                "Keep maintaining healthy habits — regular sleep, exercise, and social connection go a long way.")
    severe_any = any(x.lower() in ["severe", "extremely severe"] for x in levels)
    if severe_any:
        return ("<b>Please seek professional support.</b> One or more dimensions show a high severity level. "
                "Reach out to a licensed mental health professional or counsellor as soon as possible. "
                "You don't have to navigate this alone.")
    return ("<b>Consider speaking to someone.</b> Mild or moderate symptoms are present. "
            "Talking to a trusted friend, family member, or counsellor can be very helpful. "
            "Self-care practices like mindfulness, exercise, and adequate sleep are also beneficial.")

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.abspath(__file__))
    files = {
        'stress_model':     'stress_model.pkl',
        'anxiety_model':    'anxiety_model.pkl',
        'depression_model': 'depression_model.pkl',
        'le_stress':        'le_stress.pkl',
        'le_anxiety':       'le_anxiety.pkl',
        'le_depression':    'le_depression.pkl',
    }
    loaded = {}
    for key, fname in files.items():
        path = os.path.join(base, fname)
        if os.path.exists(path):
            loaded[key] = joblib.load(path)
        else:
            loaded[key] = None
    return loaded

models = load_models()
missing = [k for k, v in models.items() if v is None]
if missing:
    st.error(f"⚠️ Model files not found: {', '.join(missing)}. "
             f"Make sure all .pkl files are in the same folder as app.py.")
    st.stop()

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'results' not in st.session_state:
    st.session_state.results = None

# ─────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────
def page_home():
    st.markdown("""
    <div class="hero-banner">
        <h1>DASS-42 Mental Health Assessment</h1>
        <p>A research-based self-report tool to assess your<br>
        <b>Stress · Anxiety · Depression</b> levels</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        <b>Important Disclaimer:</b> This tool is for <b>informational and educational purposes only</b>.
        It does <b>not</b> provide a clinical diagnosis. Always consult a qualified mental health
        professional for medical advice, diagnosis, or treatment.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <div class="section-title">How it works</div>
        <div class="section-subtitle">Three quick sections · Takes about 5 minutes</div>
        <p style="font-size:0.9rem; color:#374151; line-height:1.9; margin:0;">
        <b>Step 1 — Stress:</b> 8 questions about tension and difficulty relaxing.<br>
        <b>Step 2 — Anxiety:</b> 8 questions about physical and mental anxiety symptoms.<br>
        <b>Step 3 — Depression:</b> 8 questions about mood, motivation and self-worth.<br><br>
        Rate each statement based on how much it applied to you <b>over the past week</b>,
        on a scale of <b>0 (not at all)</b> to <b>3 (very much)</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Start Assessment"):
        st.session_state.page = 'assessment'
        st.rerun()

# ─────────────────────────────────────────────
# PAGE: ASSESSMENT
# ─────────────────────────────────────────────
def page_assessment():
    st.markdown("""
    <div style="text-align:center; margin-bottom:20px;">
        <span style="font-size:1.3rem; font-weight:700; color:#1F2937;">Complete the Assessment</span><br>
        <span style="font-size:0.83rem; color:#6B7280;">
        Rate how much each statement applied to you <b>over the past week</b>
        </span>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Stress", "Anxiety", "Depression"])

    # ── STRESS ──
    with tab1:
        st.markdown("""
        <div class="section-title">Stress</div>
        <div class="section-subtitle">How often did these apply to you in the past week?</div>
        """, unsafe_allow_html=True)
        s_vals = {}
        for col, question in STRESS_QUESTIONS.items():
            st.markdown(f'<div class="q-label">• {question}</div>', unsafe_allow_html=True)
            choice = st.radio(" ", RATING_OPTIONS, key=f"s_{col}", index=None, label_visibility="collapsed")
            s_vals[col] = int(choice[0]) if choice else None

    # ── ANXIETY ──
    with tab2:
        st.markdown("""
        <div class="section-title">Anxiety</div>
        <div class="section-subtitle">How often did these apply to you in the past week?</div>
        """, unsafe_allow_html=True)
        a_vals = {}
        for col, question in ANXIETY_QUESTIONS.items():
            st.markdown(f'<div class="q-label">• {question}</div>', unsafe_allow_html=True)
            choice = st.radio(" ", RATING_OPTIONS, key=f"a_{col}", index=None, label_visibility="collapsed")
            a_vals[col] = int(choice[0]) if choice else None

    # ── DEPRESSION ──
    with tab3:
        st.markdown("""
        <div class="section-title">Depression</div>
        <div class="section-subtitle">How often did these apply to you in the past week?</div>
        """, unsafe_allow_html=True)
        d_vals = {}
        for col, question in DEPRESSION_QUESTIONS.items():
            st.markdown(f'<div class="q-label">• {question}</div>', unsafe_allow_html=True)
            choice = st.radio(" ", RATING_OPTIONS, key=f"d_{col}", index=None, label_visibility="collapsed")
            d_vals[col] = int(choice[0]) if choice else None

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Get My Results"):
        # Validate all answered
        s_missing = [q for q, v in s_vals.items() if v is None]
        a_missing = [q for q, v in a_vals.items() if v is None]
        d_missing = [q for q, v in d_vals.items() if v is None]

        if s_missing or a_missing or d_missing:
            unanswered = len(s_missing) + len(a_missing) + len(d_missing)
            st.error(f"Please answer all questions before submitting. {unanswered} question(s) are unanswered.")
        else:
            s_input = np.array([[s_vals[c] for c in STRESS_QUESTIONS]])
            s_pred  = models['stress_model'].predict(s_input)[0]
            s_prob  = models['stress_model'].predict_proba(s_input)[0].max()
            s_label = models['le_stress'].inverse_transform([s_pred])[0]

            a_input = np.array([[a_vals[c] for c in ANXIETY_QUESTIONS]])
            a_pred  = models['anxiety_model'].predict(a_input)[0]
            a_prob  = models['anxiety_model'].predict_proba(a_input)[0].max()
            a_label = models['le_anxiety'].inverse_transform([a_pred])[0]

            d_input = np.array([[d_vals[c] for c in DEPRESSION_QUESTIONS]])
            d_pred  = models['depression_model'].predict(d_input)[0]
            d_prob  = models['depression_model'].predict_proba(d_input)[0].max()
            d_label = models['le_depression'].inverse_transform([d_pred])[0]

            st.session_state.results = {
                'stress':     (s_label, s_prob),
                'anxiety':    (a_label, a_prob),
                'depression': (d_label, d_prob),
            }
            st.session_state.page = 'results'
            st.rerun()

# ─────────────────────────────────────────────
# PAGE: RESULTS
# ─────────────────────────────────────────────
def page_results():
    r = st.session_state.results
    s_label, s_prob = r['stress']
    a_label, a_prob = r['anxiety']
    d_label, d_prob = r['depression']

    st.markdown("""
    <div class="hero-banner" style="padding:28px 36px;">
        <h1 style="font-size:1.6rem;">Your Assessment Results</h1>
        <p>Based on your responses over the past week</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-wrapper">
        <div class="result-card {severity_css(s_label)}">
            <div class="cat">Stress</div>
            <div class="sev">{s_label}</div>
            <div class="conf">Confidence: {s_prob*100:.1f}%</div>
        </div>
        <div class="result-card {severity_css(a_label)}">
            <div class="cat">Anxiety</div>
            <div class="sev">{a_label}</div>
            <div class="conf">Confidence: {a_prob*100:.1f}%</div>
        </div>
        <div class="result-card {severity_css(d_label)}">
            <div class="cat">Depression</div>
            <div class="sev">{d_label}</div>
            <div class="conf">Confidence: {d_prob*100:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="section-card" style="margin-top:10px;">
        <div class="section-title">Recommendation</div>
        <div class="advice-box">{get_advice(s_label, a_label, d_label)}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <div class="section-title">About this Assessment</div>
        <div class="section-subtitle">Trained on DASS-42 dataset · Top-8 features per category</div>
        <p style="font-size:0.85rem; color:#374151; line-height:1.9; margin:0;">
        <b>Stress:</b> Evaluated using top 8 most influential questions from the stress subscale.<br>
        <b>Anxiety:</b> Evaluated using top 8 most influential questions from the anxiety subscale.<br>
        <b>Depression:</b> Evaluated using top 8 most influential questions from the depression subscale.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Retake Assessment"):
            st.session_state.page = 'assessment'
            st.session_state.results = None
            st.rerun()
    with col2:
        if st.button("Back to Home"):
            st.session_state.page = 'home'
            st.session_state.results = None
            st.rerun()

# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────
if st.session_state.page == 'home':
    page_home()
elif st.session_state.page == 'assessment':
    page_assessment()
elif st.session_state.page == 'results':
    page_results()
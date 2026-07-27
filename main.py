# ai_resume_talent_scout_pro_v2_clean.py
import os
import re
import io
import json
import time
import base64
import PyPDF2
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Optional PDF export
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

# -------------------------
# NLTK setup
# -------------------------
def setup_nltk():
    try:
        sw = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        sw = set(stopwords.words('english'))

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    return frozenset(sw), WordNetLemmatizer()

STOPWORDS, lemmatizer = setup_nltk()

def load_css():
    st.markdown("""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"]{
        font-family:'Poppins',sans-serif;
    }

    .stApp{
        background:#EFE9E1;
    }

    h1,h2,h3{
        color:#322D29;
        font-weight:700;
    }

    section[data-testid="stSidebar"]{
        background:#322D29;
        color:white;
    }

    section[data-testid="stSidebar"] *{
        color:white !important;
    }

    /* Upload Box */
    [data-testid="stFileUploader"]{
        border:2px dashed #AC9C8D;
        border-radius:12px;
        background:white;
        padding:15px;
    }

    /* Text Area */
    .stTextArea textarea{
        border:2px solid #AC9C8D !important;
        border-radius:10px !important;
        background:white !important;
        color:#322D29 !important;
    }

    /* Text Input */
    .stTextInput input{
        border:2px solid #AC9C8D !important;
        border-radius:10px !important;
        color:#322D29 !important;
    }

    /* Buttons */
    .stButton>button{
        background:#72383D;
        color:white;
        border:none;
        border-radius:10px;
        font-weight:600;
        width:100%;
        transition:0.3s;
    }

    .stButton>button:hover{
        background:#5D2D31;
    }

    /* Download Button */
    .stDownloadButton>button{
        background:#322D29;
        color:white;
        border-radius:10px;
        border:none;
        width:100%;
    }

    .stDownloadButton>button:hover{
        background:#1f1b19;
    }

    /* Metric Cards */
    div[data-testid="metric-container"]{
        background:white;
        border:1px solid #D1C7BD;
        border-radius:12px;
        box-shadow:0 2px 8px rgba(0,0,0,0.08);
    }

    /* Progress */
    .stProgress > div > div{
        background:#72383D;
    }

    footer{
        visibility:hidden;
    }

    #MainMenu{
        visibility:hidden;
    }

    </style>
    """, unsafe_allow_html=True)
# -------------------------
# Skill lists
# -------------------------
PREMIER_SKILLS = {
    "Data Science": ["python", "r", "sql", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
                     "machine learning", "deep learning", "nlp", "statistics"],
    "Engineering": ["java", "c++", "javascript", "react", "node.js", "docker", "aws", "kubernetes", "git"],
    "Business": ["excel", "tableau", "power bi", "communication", "leadership"],
}
ALL_SKILLS = [s for skills in PREMIER_SKILLS.values() for s in skills]

# -------------------------
# Text utilities
# -------------------------
def extract_text_from_pdf(file):
    text = ""
    try:
        file.seek(0)
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception:
        return ""
    return text

def preprocess(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = []
    for w in text.split():
        if w not in STOPWORDS:
            tokens.append(lemmatizer.lemmatize(w))
    return " ".join(tokens)

def advanced_tfidf_similarity(resume: str, job: str):
    resume = resume or ""
    job = job or ""
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=7000)
        tfidf = vectorizer.fit_transform([resume, job])
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return round(score * 100, 2), vectorizer, tfidf
    except Exception:
        return 0.0, None, None

def extract_skills(text: str):
    text = (text or "").lower()
    found = [skill for skill in ALL_SKILLS if skill in text]
    return found

def compare_skills(resume_text: str, job_text: str):
    job = set(extract_skills(job_text))
    res = set(extract_skills(resume_text))
    return sorted(list(job & res)), sorted(list(job - res))

def generate_ai_insights(overall, tfidf, missing_skills, missing_kw):
    insights = []
    if overall >= 85:
        insights.append("⭐ Excellent match! Very strong fit.")
    elif overall >= 70:
        insights.append("👍 Good match. Minor improvements can increase score.")
    else:
        insights.append("⚠ Improve resume alignment with job keywords.")
    if missing_skills:
        insights += [f"💡 Add or highlight experience related to {s}" for s in missing_skills[:5]]
    if missing_kw:
        insights.append("📝 Add keywords: " + ", ".join(missing_kw[:10]))
    return insights

def create_report(name, tfidf, skill, overall, matched_kw, missing_kw, matched_sk, missing_sk, insights, filename):
    return {
        "candidate": name,
        "file": filename,
        "tfidf_score": tfidf,
        "skill_match_percent": skill,
        "overall_score": overall,
        "matched_keywords": matched_kw,
        "missing_keywords": missing_kw,
        "matched_skills": matched_sk,
        "missing_skills": missing_sk,
        "ai_insights": insights,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

# -------------------------
# Visualization helpers
# -------------------------
def plot_skill_radar(matched_skills, missing_skills):
    categories = list(set(matched_skills + missing_skills))
    if not categories:
        return None
    matched_vals = [1 if s in matched_skills else 0.2 for s in categories]
    categories += [categories[0]]
    matched_vals += [matched_vals[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=matched_vals, theta=categories, fill='toself', name='Skill presence'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=False, template="plotly_dark", margin=dict(l=20,r=20,t=30,b=20))
    return fig

def keyword_heatmap(jd_tokens, resume_tokens):
    jd_freq = {}
    for t in jd_tokens:
        jd_freq[t] = jd_freq.get(t, 0) + 1
    top = sorted(jd_freq.items(), key=lambda x: -x[1])[:20]
    words = [w for w, _ in top]
    presence = [1 if w in resume_tokens else 0 for w in words]
    fig = go.Figure(data=go.Bar(x=words, y=presence))
    fig.update_layout(template="plotly_dark", yaxis=dict(tickvals=[0,1], ticktext=["Missing","Present"]), margin=dict(t=10,b=10))
    return fig

# -------------------------
# Streamlit app
# -------------------------
def main():
    st.set_page_config(page_title="AI Resume Matcher", layout="wide", page_icon="🧾")
    load_css()
    st.title("AI Resume Matcher — Talent Scout Pro v2.0")

    # Sidebar settings (minimal / useful)
    with st.sidebar:
        st.header("Settings")
        candidate_prefix = st.text_input("Candidate name prefix", "Candidate")
        allow_multiple = st.checkbox("Allow multiple resume upload", value=True)
        st.markdown("---")
        if FPDF_AVAILABLE:
            st.markdown("PDF export: ✅")
        else:
            st.markdown("PDF export: ❌)")

    # Upload area
    st.header("Upload resumes and paste Job Description")
    resumes = st.file_uploader("Choose resume PDF(s)", type=["pdf"], accept_multiple_files=allow_multiple)
    jd = st.text_area("Paste Job Description here", height=240)

    run = st.button("Run Analysis")
    if run:
        if not resumes or not jd.strip():
            st.error("Please upload at least one resume PDF and paste the job description.")
            return

        jd_proc = preprocess(jd)
        results = []
        resume_files = resumes if isinstance(resumes, list) else [resumes]
        progress = st.progress(0)
        total = len(resume_files)

        for i, f in enumerate(resume_files, start=1):
            raw_text = extract_text_from_pdf(f)
            proc_text = preprocess(raw_text)
            tfidf_score, _, _ = advanced_tfidf_similarity(proc_text, jd_proc)

            resume_tokens = set(proc_text.split())
            jd_tokens = set(jd_proc.split())

            matched_kw = sorted(list(resume_tokens & jd_tokens))
            missing_kw = sorted(list(jd_tokens - resume_tokens))

            matched_sk, missing_sk = compare_skills(proc_text, jd_proc)

            skill_score = round(len(matched_sk) / max(1, len(matched_sk + missing_sk)) * 100, 2)
            overall = round(0.6 * tfidf_score + 0.4 * skill_score, 2)

            insights = generate_ai_insights(overall, tfidf_score, missing_sk, missing_kw)

            report = create_report(candidate_prefix, tfidf_score, skill_score, overall,
                                   matched_kw, missing_kw, matched_sk, missing_sk, insights, f.name)

            results.append({"file": f.name, "report": report, "raw_text": raw_text, "proc_text": proc_text})
            progress.progress(int(i/total * 100))

        progress.empty()
        st.success("Analysis completed — see results below.")

        # Display results
        for item in results:
            rep = item["report"]
            st.markdown("---")
            st.subheader(f"{item['file']} — Overall: {rep['overall_score']}%")
            st.write(f"**TF-IDF Score:** {rep['tfidf_score']}%   |   **Skill Match:** {rep['skill_match_percent']}%")

            st.markdown("**Top insights**")
            for insight in rep["ai_insights"]:
                st.write(f"- {insight}")

            # Matched / missing skills
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Matched Skills**")
                st.write(", ".join(rep["matched_skills"]) if rep["matched_skills"] else "_None detected_")
            with c2:
                st.markdown("**Missing Skills (from JD)**")
                st.write(", ".join(rep["missing_skills"]) if rep["missing_skills"] else "_None detected_")

            # Keyword heatmap
            fig_hm = keyword_heatmap(set(jd_proc.split()), set(item["proc_text"].split()))
            if fig_hm:
                st.plotly_chart(fig_hm, use_container_width=True)

            # Skill radar
            radar = plot_skill_radar(rep["matched_skills"], rep["missing_skills"])
            if radar:
                st.plotly_chart(radar, use_container_width=True)

            # Raw & processed toggles
            with st.expander("Show extracted resume text (raw)"):
                st.code(item["raw_text"][:10000] + ("..." if len(item["raw_text"]) > 10000 else ""))
            with st.expander("Show preprocessed tokens (top 200)"):
                st.write(" ".join(item["proc_text"].split()[:200]))

            # Downloads
            json_bytes = json.dumps(rep, indent=2).encode()
            st.download_button("⬇️ Download JSON Report", data=json_bytes, file_name=f"{candidate_prefix}_{item['file']}_report.json", mime="application/json")

            if FPDF_AVAILABLE:
                try:
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.cell(0, 10, txt=f"Candidate: {rep['candidate']}", ln=1)
                    pdf.cell(0, 10, txt=f"File: {rep['file']}", ln=1)
                    pdf.cell(0, 10, txt=f"Overall Score: {rep['overall_score']}%", ln=1)
                    pdf.ln(4)
                    pdf.multi_cell(0, 6, txt="Insights:")
                    for insight in rep["ai_insights"]:
                        pdf.multi_cell(0, 6, txt="- " + insight)
                    pdf_bytes = pdf.output(dest='S').encode('latin-1')
                    st.download_button("⬇️ Download PDF Report", data=pdf_bytes, file_name=f"{candidate_prefix}_{item['file']}_report.pdf", mime="application/pdf")
                except Exception as e:
                    st.warning("PDF export failed: " + str(e))

    # end run

if __name__ == "__main__":
    main()


# Enhanced version placeholder: integrate additional matplotlib/seaborn charts as discussed.

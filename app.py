import streamlit as st
import PyPDF2
import spacy
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========= ONE-TIME DOWNLOADS (first run only) =========
nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

# ========= SIMPLE SKILL KEYWORD LIST =========
# You can expand this list anytime
SKILL_KEYWORDS = {
    "python", "java", "c++", "c#", "javascript", "typescript",
    "html", "css", "react", "angular", "vue","dsa","dbms",
    "django", "flask", "spring", "spring boot", "node", "node.js",
    "sql", "mysql", "postgresql", "mongodb",
    "machine learning", "deep learning", "nlp",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "data analysis", "data science", "power bi", "tableau",
    "git", "docker", "kubernetes", "aws", "azure", "gcp",

}


# ========= HELPER FUNCTIONS =========
def extract_resume_text_from_file(uploaded_file):
    """Read PDF from Streamlit UploadedFile and return full text."""
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def preprocess_text(text: str) -> str:
    """Lowercase + remove stopwords & punctuation using spaCy + NLTK."""
    text = text.lower()
    doc = nlp(text)

    clean_tokens = [
        token.text
        for token in doc
        if token.text not in stop_words and token.text not in string.punctuation
    ]
    return " ".join(clean_tokens)


def compute_similarity_scores(resume_texts, job_description_text):
    """
    resume_texts: list of cleaned resume strings
    job_description_text: cleaned JD string
    Returns: list of similarity scores (0–1)
    """
    corpus = [job_description_text] + resume_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(jd_vector, resume_vectors).flatten()
    return similarities


def extract_skills_from_text(text: str):
    """
    Very simple skill extractor:
    - Look for known SKILL_KEYWORDS inside the text.
    Returns a set of matched skills.
    """
    text_low = text.lower()
    found = set()
    for skill in SKILL_KEYWORDS:
        if skill.lower() in text_low:
            found.add(skill)
    return found


# ========= BEAUTIFUL STREAMLIT UI =========
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide",
)

# Top header
st.markdown(
    """
    <h1 style="text-align:center; color:#2d6cdf;">
        📄 AI-Based Resume Classifier & Job Matching System
    </h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style="text-align:center; color:gray;">
        Upload multiple resumes, paste a job description, and let the AI rank candidates,
        <b>show matching skills and missing skills</b>.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Layout: left = inputs, right = explanation
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("1️⃣ Upload Resumes")
    uploaded_resumes = st.file_uploader(
        "Upload one or more resume PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can select multiple PDF files here.",
    )

    st.subheader("2️⃣ Paste Job Description (JD)")
    job_description = st.text_area(
        "",
        height=220,
        placeholder=(
            "Example: We are looking for a Data Scientist with 3+ years of experience "
            "in Python, Machine Learning, SQL, and data analysis..."
        ),
    )

    analyze_button = st.button("🚀 Analyze Resumes", use_container_width=True)

with right_col:
    st.subheader("ℹ What this tool does")
    st.write(
        """
        - Reads each resume (PDF) and cleans the text  
        - Cleans the Job Description (JD)  
        - Uses *TF-IDF + cosine similarity* to compute a match score  
        - Extracts *skills mentioned in the JD*  
        - For every resume, shows:
          - ✅ *Matched skills*
          - ⚠ *Missing skills* (present in JD but not found in resume)
        """
    )
    st.info(
        "Tip: Make your JD skill-rich (e.g., Python, SQL, Machine Learning, AWS) "
        "so the matching works better."
    )

st.markdown("---")

# ========= MAIN LOGIC =========
if analyze_button:
    if not uploaded_resumes:
        st.error("❌ Please upload at least one resume PDF.")
    elif not job_description.strip():
        st.error("❌ Please paste a Job Description.")
    else:
        with st.spinner("Processing resumes... this may take a few seconds ⏳"):
            # RAW texts (for skill extraction)
            raw_jd = job_description
            cleaned_jd = preprocess_text(raw_jd)

            # Skills required based on JD
            jd_skills = extract_skills_from_text(raw_jd)

            resume_results = []
            cleaned_resumes = []
            resume_names = []
            raw_resume_texts = []

            # Extract + preprocess each resume
            for uploaded_file in uploaded_resumes:
                resume_names.append(uploaded_file.name)

                uploaded_file.seek(0)
                raw_text = extract_resume_text_from_file(uploaded_file)
                raw_resume_texts.append(raw_text)

                cleaned_text = preprocess_text(raw_text)
                cleaned_resumes.append(cleaned_text)

            # Similarity scores (0–1)
            similarity_scores = compute_similarity_scores(cleaned_resumes, cleaned_jd)

            # Build results with skills
            for name, score, raw_text in zip(
                resume_names, similarity_scores, raw_resume_texts
            ):
                resume_skills = extract_skills_from_text(raw_text)

                matched_skills = sorted(jd_skills & resume_skills)
                missing_skills = sorted(jd_skills - resume_skills)

                resume_results.append(
                    {
                        "Resume File": name,
                        "Similarity Score (0–1)": round(float(score), 3),
                        "Similarity (%)": round(float(score) * 100, 1),
                        "Matched Skills": ", ".join(matched_skills)
                        if matched_skills
                        else "-",
                        "Missing Skills": ", ".join(missing_skills)
                        if missing_skills
                        else "-",
                    }
                )

        # ========= DISPLAY RESULTS =========
        st.subheader("📊 Overall Results")

        # Sort by highest similarity
        resume_results = sorted(
            resume_results, key=lambda x: x["Similarity Score (0–1)"], reverse=True
        )

        # Show top candidate summary
        best = resume_results[0]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Top Candidate", best["Resume File"])
        with c2:
            st.metric("Best Match (%)", best["Similarity (%)"])
        with c3:
            st.metric("JD Skills Count", len(jd_skills))

        st.markdown("### 🧾 Detailed Table")
        st.dataframe(resume_results, use_container_width=True)

        # Per-resume expandable details
        st.markdown("### 🔍 Per-Resume Skill Details")
        for res in resume_results:
            with st.expander(f"📄 {res['Resume File']}"):
                st.write(f"*Similarity:* {res['Similarity (%)']} %")
                st.write(f"*Matched Skills:* {res['Matched Skills']}")
                st.write(f"*Missing Skills:* {res['Missing Skills']}")

        st.success("✅ Analysis complete! Scroll up to see the ranking and skill breakdown.")
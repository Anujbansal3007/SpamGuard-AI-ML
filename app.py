import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT /  "sms_spam.csv"

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Upload data/sms_spam.csv to your GitHub repository."
        )

    df = pd.read_csv(DATA_PATH)
    return df.dropna(subset=["text", "label"])

st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# Neon glass UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --bg:#070918; --panel:#10152d; --line:rgba(129,147,255,.22); --cyan:#22e6ff; --purple:#b34cff; --green:#21f6b5; }
.stApp { background: radial-gradient(ellipse at 76% 2%, rgba(91,31,181,.22), transparent 34%), radial-gradient(ellipse at 15% 25%, rgba(0,201,255,.09), transparent 30%), #070918; color:#f5f6ff; font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0d1027,#080a19); border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:#e7eaff; }
h1,h2,h3,.hero-title { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.04em; }
.hero-title { font-size: clamp(2.3rem, 5vw, 4.2rem); line-height:1.02; font-weight:700; margin:0; }
.gradient-text { background:linear-gradient(90deg,#d24cff,#24dfff 65%,#3ef9c2); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.muted { color:#a6afd8; }
.eyebrow { display:inline-block; padding:7px 12px; border:1px solid rgba(179,76,255,.45); background:rgba(135,57,255,.12); border-radius:999px; color:#d9c2ff; font-size:.82rem; margin-bottom:16px; }
.glass { background:linear-gradient(145deg,rgba(22,29,62,.91),rgba(10,14,34,.93)); border:1px solid var(--line); border-radius:20px; padding:22px; box-shadow:0 12px 40px rgba(0,0,0,.18); }
.stat { background:linear-gradient(145deg,rgba(23,31,68,.88),rgba(11,14,34,.95)); border:1px solid var(--line); border-radius:17px; padding:18px 20px; min-height:112px; }
.stat-label { color:#a8b2dd; font-size:.85rem; }
.stat-value { font-family:'Space Grotesk'; font-size:1.8rem; font-weight:700; margin-top:6px; }
.stat-icon { font-size:1.3rem; }
div.stButton > button { border:1px solid rgba(90,216,255,.6); color:white; background:linear-gradient(100deg,#a82dff,#5d49ff 55%,#00bce8); border-radius:12px; font-weight:700; min-height:47px; box-shadow:0 0 24px rgba(115,58,255,.22); }
div.stButton > button:hover { border-color:#7ef4ff; transform:translateY(-1px); }
.stTextArea textarea, .stTextInput input { background:#0b1027 !important; color:#f6f7ff !important; border:1px solid rgba(113,139,255,.36) !important; border-radius:14px !important; }
div[data-testid="stFileUploader"] { background:rgba(11,16,39,.6); border:1px dashed rgba(89,211,255,.4); border-radius:14px; padding:10px; }
hr { border-color:rgba(129,147,255,.16); }
footer {visibility:hidden;}
#MainMenu {visibility:hidden;}
div[data-testid="stMetric"] { background:rgba(14,20,47,.75); padding:14px; border:1px solid var(--line); border-radius:14px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH).dropna(subset=["text", "label"])

@st.cache_resource
def train_model():
    df = load_data()
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))
    ])
    model.fit(df["text"].astype(str), df["label"].astype(str))
    return model, df

def clean_message(text):
    return re.sub(r"\s+", " ", str(text)).strip()

def prediction_details(model, message):
    pred = model.predict([message])[0]
    probs = model.predict_proba([message])[0]
    classes = list(model.classes_)
    confidence = float(probs[classes.index(pred)]) * 100
    return pred, confidence

model, data = train_model()
total = len(data)
spam_n = int((data["label"].str.lower() == "spam").sum())
ham_n = total - spam_n

with st.sidebar:
    st.markdown("<div style='font-size:2rem'>🛡️</div><div style='font-family:Space Grotesk;font-size:1.35rem;font-weight:700'>Spam<span style='color:#26e6ff'>Guard</span> <span style='color:#b34cff'>AI</span></div><div style='font-size:.8rem;color:#9fa9d5;margin-bottom:20px'>INTELLIGENT MESSAGE SECURITY</div>", unsafe_allow_html=True)
    page = st.radio("NAVIGATION", ["⚡ Dashboard", "🔎 Detect Spam", "📊 Analytics", "🧠 Model Lab", "ℹ️ About"], label_visibility="visible")
    st.markdown("---")
    st.markdown("**SYSTEM STATUS**")
    st.markdown("🟢 Model online")
    st.markdown("🔐 Local inference")
    st.markdown(f"<div style='font-size:.8rem;color:#929dc9;margin-top:20px'>Training examples<br><b style='font-size:1.4rem;color:white'>{total}</b><br><br>Built with Python + scikit-learn</div>", unsafe_allow_html=True)

if page == "⚡ Dashboard":
    st.markdown('<div class="eyebrow">✦ MACHINE LEARNING • MESSAGE INTELLIGENCE</div>', unsafe_allow_html=True)
    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        st.markdown('<div class="hero-title">Detect spam.<br><span class="gradient-text">Keep your inbox safe.</span></div>', unsafe_allow_html=True)
        st.markdown('<p class="muted" style="font-size:1.08rem;line-height:1.7;margin-top:18px">A machine-learning message scanner that spots suspicious language, scam offers, and unwanted messages in seconds.</p>', unsafe_allow_html=True)
        if st.button("🚀  Analyze a message", use_container_width=True):
            st.session_state["nav_jump"] = True
            st.info("Choose **🔎 Detect Spam** from the sidebar to run the scanner.")
    with right:
        st.markdown("""
        <div class="glass" style="min-height:240px;display:flex;flex-direction:column;justify-content:center">
          <div style="font-size:4.4rem;text-align:center;filter:drop-shadow(0 0 20px #8e46ff)">🛡️</div>
          <div style="font-family:Space Grotesk;font-size:1.35rem;text-align:center;font-weight:700">AI-powered protection</div>
          <div class="muted" style="text-align:center;margin-top:8px">Text patterns • Risk signals • Fast predictions</div>
          <div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-top:20px">
            <span style="border:1px solid #ff547a;color:#ff8da4;border-radius:20px;padding:6px 10px;font-size:.8rem">🚨 Spam alert</span>
            <span style="border:1px solid #1de6b0;color:#70ffd6;border-radius:20px;padding:6px 10px;font-size:.8rem">✓ Safe message</span>
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(4)
    stats = [("🧾","Training messages",f"{total}","Curated sample dataset"),("🚨","Spam examples",f"{spam_n}","Labeled spam samples"),("🟢","Legitimate examples",f"{ham_n}","Labeled safe samples"),("⚙️","Model","TF-IDF + LR","Text classification")]
    for c,(icon,label,value,sub) in zip(cols,stats):
        with c:
            st.markdown(f'<div class="stat"><div class="stat-icon">{icon}</div><div class="stat-label">{label}</div><div class="stat-value">{value}</div><div class="stat-label" style="margin-top:4px;font-size:.75rem">{sub}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="glass"><h3 style="margin-top:0">⚡ Quick test</h3><div class="muted">Try a sample message to see the classifier in action.</div></div>', unsafe_allow_html=True)
    sample = st.selectbox("Choose a sample", [
        "Congratulations! You won a free prize. Claim now by clicking this link.",
        "Hey, are we still meeting for lunch at 1 pm?",
        "URGENT! Your account is suspended. Verify your password immediately.",
        "Thanks for sending the notes, I will review them tonight."
    ])
    if st.button("Run quick scan", key="quick_scan"):
        label, conf = prediction_details(model, sample)
        is_spam = label.lower() == "spam"
        if is_spam:
            st.error(f"🚨 Likely SPAM · model score {conf:.1f}%")
        else:
            st.success(f"🟢 Likely NOT SPAM · model score {conf:.1f}%")
        st.caption("This score is the model's estimated class probability, not a guarantee.")

elif page == "🔎 Detect Spam":
    st.markdown('<div class="eyebrow">🔎 LIVE MESSAGE SCANNER</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title" style="font-size:3rem">Check your <span class="gradient-text">message.</span></div>', unsafe_allow_html=True)
    st.markdown('<p class="muted">Paste an SMS, email, or suspicious text. Your message is analyzed locally by the trained model.</p>', unsafe_allow_html=True)
    a,b = st.columns([1.35,.65], gap="large")
    with a:
        message = st.text_area("MESSAGE INPUT", height=220, max_chars=5000, placeholder="Paste your message here...\n\nExample: Congratulations! You have won a cash prize. Click here to claim.")
        x,y = st.columns([1,1])
        with x:
            analyze = st.button("🔍  Analyze message", use_container_width=True)
        with y:
            if st.button("↻  Clear example", use_container_width=True):
                st.rerun()
        if analyze:
            message = clean_message(message)
            if len(message) < 3:
                st.warning("Enter a message with at least 3 characters.")
            else:
                label, conf = prediction_details(model, message)
                is_spam = label.lower() == "spam"
                st.session_state["last_result"] = {"message": message, "label": label, "confidence": conf}
                st.markdown("<br>", unsafe_allow_html=True)
                if is_spam:
                    st.error("🚨 SPAM DETECTED — treat this message with caution.")
                else:
                    st.success("🟢 LIKELY LEGITIMATE — no strong spam pattern detected.")
                st.progress(min(max(int(conf),0),100), text=f"Model confidence score: {conf:.1f}%")
                st.markdown("**What to do next**")
                if is_spam:
                    st.markdown("- Avoid opening links or attachments.\n- Never share OTPs, passwords, or bank details.\n- Verify the sender through an official channel.")
                else:
                    st.markdown("- Still verify unexpected links and requests.\n- A legitimate-looking message can also be malicious.")
                st.caption("Machine-learning predictions can be wrong. Do not treat this as a security guarantee.")
    with b:
        st.markdown("""<div class="glass">
        <div style="font-size:2rem">🧬</div><h3>How it works</h3>
        <p class="muted">01 · Text is converted into TF-IDF features.</p>
        <p class="muted">02 · Logistic Regression scores the message.</p>
        <p class="muted">03 · The predicted class and score appear here.</p>
        <hr><div style="color:#21f6b5;font-weight:700">● Privacy-first demo</div>
        <div class="muted" style="font-size:.86rem">Messages are processed in this app session and are not sent to an external prediction API.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="glass"><h4 style="margin-top:0">Try these signals</h4><span style="color:#ff829d">🚩 Urgency</span><br><span class="muted">“Act now”, “account blocked”</span><br><br><span style="color:#ffca75">💸 Prize or money bait</span><br><span class="muted">“You won”, “claim cash”</span><br><br><span style="color:#5fe8ff">🔗 Suspicious links</span><br><span class="muted">Unexpected links or requests</span></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Batch scan a CSV")
    st.caption("Upload a CSV containing a column named `text` or `message`. The app adds predicted labels and confidence scores.")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="batch")
    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            col = "text" if "text" in batch_df.columns else ("message" if "message" in batch_df.columns else None)
            if not col:
                st.error("CSV needs a column named 'text' or 'message'.")
            else:
                preds = model.predict(batch_df[col].fillna("").astype(str))
                probs = model.predict_proba(batch_df[col].fillna("").astype(str))
                classes = list(model.classes_)
                batch_df["prediction"] = preds
                batch_df["confidence_%"] = [round(float(p[classes.index(lbl)])*100, 1) for p,lbl in zip(probs,preds)]
                st.dataframe(batch_df, use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download predictions CSV", batch_df.to_csv(index=False).encode("utf-8"), "spamguard_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not read this CSV: {e}")

elif page == "📊 Analytics":
    st.markdown('<div class="eyebrow">📊 MODEL INSIGHTS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title" style="font-size:3rem">Data <span class="gradient-text">overview.</span></div>', unsafe_allow_html=True)
    st.markdown('<p class="muted">Explore the small, bundled demonstration dataset used to train this project.</p>', unsafe_allow_html=True)
    counts = data["label"].str.lower().value_counts()
    c1,c2,c3 = st.columns(3)
    c1.metric("Total examples", len(data))
    c2.metric("Spam messages", int(counts.get("spam",0)))
    c3.metric("Legitimate messages", int(counts.get("ham",0)))
    left,right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="glass"><h3>Label distribution</h3>', unsafe_allow_html=True)
        chart_df = counts.rename_axis("Class").reset_index(name="Messages")
        st.bar_chart(chart_df.set_index("Class"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="glass"><h3>Example messages</h3>', unsafe_allow_html=True)
        st.dataframe(data.sample(min(8,len(data)), random_state=7).reset_index(drop=True), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.info("This is a compact demo dataset for learning and presentation. Results on real-world messages may differ; use a larger, representative labeled dataset for serious evaluation.")

elif page == "🧠 Model Lab":
    st.markdown('<div class="eyebrow">🧠 INSIDE THE CLASSIFIER</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title" style="font-size:3rem">The model <span class="gradient-text">lab.</span></div>', unsafe_allow_html=True)
    st.markdown("### Pipeline architecture")
    st.markdown("""
    <div class="glass">
      <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;justify-content:center">
        <div style="border:1px solid #2acfff;border-radius:12px;padding:15px;text-align:center;min-width:130px"><b>Raw text</b><br><span class="muted">SMS / email</span></div>
        <div style="color:#b34cff;font-size:1.5rem">→</div>
        <div style="border:1px solid #b34cff;border-radius:12px;padding:15px;text-align:center;min-width:130px"><b>TF-IDF</b><br><span class="muted">Word n-grams</span></div>
        <div style="color:#b34cff;font-size:1.5rem">→</div>
        <div style="border:1px solid #21f6b5;border-radius:12px;padding:15px;text-align:center;min-width:130px"><b>Logistic Reg.</b><br><span class="muted">Classifier</span></div>
        <div style="color:#b34cff;font-size:1.5rem">→</div>
        <div style="border:1px solid #ffca75;border-radius:12px;padding:15px;text-align:center;min-width:130px"><b>Prediction</b><br><span class="muted">Spam / Ham</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Current configuration")
    st.dataframe(pd.DataFrame([
        ["Feature extraction","TF-IDF word unigrams and bigrams"],
        ["Classifier","Logistic Regression"],
        ["Class balancing","Balanced class weights"],
        ["Training samples",str(total)],
        ["Labels","spam / ham"],
        ["External API","Not required"]
    ], columns=["Setting","Value"]), use_container_width=True, hide_index=True)
    st.markdown("### Evaluate on a held-out split")
    if st.button("Run evaluation (80/20 split)"):
        X_train, X_test, y_train, y_test = train_test_split(data["text"].astype(str), data["label"].astype(str), test_size=0.2, random_state=42, stratify=data["label"])
        eval_model = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1,2), sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))
        ])
        eval_model.fit(X_train,y_train)
        yp = eval_model.predict(X_test)
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Accuracy",f"{accuracy_score(y_test,yp)*100:.1f}%")
        m2.metric("Precision",f"{precision_score(y_test,yp,pos_label='spam',zero_division=0)*100:.1f}%")
        m3.metric("Recall",f"{recall_score(y_test,yp,pos_label='spam',zero_division=0)*100:.1f}%")
        m4.metric("F1 score",f"{f1_score(y_test,yp,pos_label='spam',zero_division=0)*100:.1f}%")
        st.caption("Small demo dataset: these metrics are educational and are not evidence of production-grade performance.")

else:
    st.markdown('<div class="eyebrow">ℹ️ PROJECT DETAILS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title" style="font-size:3rem">Built for safer <span class="gradient-text">messaging.</span></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass">
    <h3>SpamGuard AI — ML mini project</h3>
    <p class="muted">A student-friendly Natural Language Processing project that classifies text messages as spam or legitimate (ham).</p>
    <hr>
    <b>Tech stack</b><br>Python · Streamlit · pandas · scikit-learn · TF-IDF · Logistic Regression
    <br><br><b>Project features</b><br>• Interactive message scanner<br>• Confidence score visualization<br>• Batch CSV classification and export<br>• Dataset analytics<br>• Train/test evaluation metrics<br>• Responsive neon-glass dashboard
    <br><br><b>Important limitation</b><br>The included dataset is small and illustrative. For a credible final-year or portfolio project, replace it with a larger public SMS spam dataset, evaluate on unseen data, and report its source and limitations.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Run locally")
    st.code("python -m venv .venv\n.venv\\Scripts\\activate\npython -m pip install -r requirements.txt\npython -m streamlit run app.py", language="powershell")

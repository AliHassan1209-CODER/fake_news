import streamlit as st
import pickle
import numpy as np
import pandas as pd
from streamlit.components.v1 import html

# Page settings
st.set_page_config(page_title="🧠 Fake News Detector", layout="centered", page_icon="📰")

# Background + styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-image: url('https://images.unsplash.com/photo-1555066931-4365d14bab8c');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #000;
    }

    .stApp {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        margin: auto;
    }

    .stButton>button {
        background-color: #5C6BC0;
        color: white;
        padding: 0.6em 2em;
        font-size: 16px;
        border: none;
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #3949AB;
    }

    .result-box {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 20px;
        text-align: center;
    }

    .title {
        font-size: 36px;
        text-align: center;
        font-weight: bold;
        color: #2C3E50;
        margin-top: 20px;
    }

    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #555;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Fade-in animation
html("""<script>
document.body.style.opacity = 0;
setTimeout(function(){ document.body.style.transition = "opacity 1s"; document.body.style.opacity = 1; }, 100);
</script>""", height=0)

# Load model/vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Session history
if "history" not in st.session_state:
    st.session_state.history = []

# Title
st.markdown('<div class="title">🧠 Fake News Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Paste news content or upload a file to detect misinformation</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📤 Upload a .txt file", type=["txt"])

if uploaded_file:
    file_content = uploaded_file.read().decode("utf-8")
    st.text_area("📄 File Content", value=file_content, height=150, key="file_input")
    st.session_state.file_news = file_content
else:
    st.session_state.file_news = None

# Manual input
if "title_input" not in st.session_state:
    st.session_state["title_input"] = ""

if "body_input" not in st.session_state:
    st.session_state["body_input"] = ""

if "file_input" not in st.session_state:
    st.session_state["file_input"] = ""

title = st.text_input("📰 News Title", value=st.session_state.title_input, key="title_input")
body = st.text_area("✏️ News Body", value=st.session_state.body_input, key="body_input")


# Final news
news = ""
if st.session_state.file_news:
    news = st.session_state.file_news
elif title or body:
    news = title + " " + body

# Buttons
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    check = st.button("Detect")

with col2:
    clear_fields = st.button("Clear Fields")

if clear_fields:
    st.session_state.update({
        "title_input": "",
        "body_input": "",
        "file_input": "",
        "file_news": None
    })
    st.rerun()



with col3:
    clear_history = st.button("Clear History")

with col4:
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "prediction_history.csv", "text/csv")

# Prediction
if check and news.strip():
    vec = vectorizer.transform([news])
    prediction = model.predict(vec)
    proba = model.decision_function(vec)
    confidence = np.round(abs(proba[0]) / max(abs(proba[0]), 1) * 100, 2)
    result = "REAL ✅" if prediction[0] == 1 else "FAKE ❌"

    # Add to session history
    st.session_state.history.append({
        "Title": title[:40] + ("..." if len(title) > 40 else ""),
        "Prediction": result,
        "Confidence": f"{confidence}%",
    })

    # Show result
    st.markdown(f"""
        <div class="result-box">
            <h3>Prediction: <span style='color: {"green" if prediction[0]==1 else "red"}'>{result}</span></h3>
            <p><b>Confidence:</b> {confidence}%</p>
        </div>
    """, unsafe_allow_html=True)

    st.progress(int(confidence))

elif check and not news.strip():
    st.warning("Please enter news text or upload a file.")

elif clear_history:
    st.session_state.history = []
    st.rerun()

# History
if st.session_state.history:
    st.markdown("### 🕓 Prediction History")
    st.table(st.session_state.history)

# Info
with st.expander("ℹ️ About This App"):
    st.info("""
    This AI-powered app detects fake news using a trained machine learning model.
    
    ✳️ Features:
    - Title & body input
    - Text file upload
    - Confidence meter
    - History table
    - CSV download
    
    Developed by Ali Hassan 💻
    """)

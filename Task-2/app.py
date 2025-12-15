import streamlit as st
import os
import requests
import pandas as pd
from datetime import datetime
from datasets import load_dataset, Dataset
from huggingface_hub import login

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="AI Feedback System", layout="wide")

# --------------------------------------------------
# Light UI Styling
# --------------------------------------------------
st.markdown("""
<style>
.stApp { background-color: #ffffff; color: #262730; }
section[data-testid="stSidebar"] {
    background-color: #f8f9fa !important;
    border-right: 1px solid #e5e7eb;
}
section[data-testid="stSidebar"] * { color: #262730 !important; }
.main-title {
    font-size: 42px; font-weight: 800; text-align: center;
}
.sub-text {
    text-align: center; font-size: 16px;
    color: #6b7280; margin-bottom: 20px;
}
.card {
    background-color: #ffffff; padding: 24px;
    border-radius: 16px; border: 1px solid #e5e7eb;
    box-shadow: 0 4px 16px rgba(0,0,0,0.05);
}
.stButton>button {
    background-color: #ff4b4b; color: white;
    border-radius: 10px; padding: 10px 20px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("Dashboard")
page = st.sidebar.radio("Go to", ["User Dashboard", "Admin Dashboard"])

# --------------------------------------------------
# Secrets
# --------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

# --------------------------------------------------
# Dataset
# --------------------------------------------------
DATASET_NAME = "Kirtan111/ai-feedback-data"

def load_data():
    try:
        ds = load_dataset(
            DATASET_NAME,
            split="train",
            download_mode="force_redownload"
        )
        return ds.to_list()
    except Exception as e:
        st.error(f"Dataset load failed: {e}")
        return []

def save_all(data):
    Dataset.from_list(data).push_to_hub(DATASET_NAME)

def save_entry(entry):
    data = load_data()
    data.append(entry)
    save_all(data)


import re

def clean_llm_output(text: str) -> str:
    if not text:
        return ""

    # Remove common LLM control tokens
    patterns = [
        r"<s>", r"</s>",
        r"\[OUT\]", r"\[/OUT\]",
        r"<\|assistant\|>",
        r"<\|endoftext\|>",
    ]

    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

# --------------------------------------------------
# LLM
# --------------------------------------------------
def call_llm(prompt, model="mistralai/mistral-7b-instruct"):
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            },
            timeout=30
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
        return clean_llm_output(raw)

    except Exception as e:
        return ""

# --------------------------------------------------
# Prompts (FIXED)
# --------------------------------------------------
def user_prompt(item, rating, review):
    return f"""
You are a polite and empathetic customer support assistant.
Respond in 2–3 friendly sentences.

Item: {item}
Rating: {rating}
Review: {review}
"""

def summary_prompt(review):
    return f"""
You are an analyst.
Write EXACTLY ONE complete sentence summarizing the customer feedback below.
Do NOT leave the response empty.

Customer review:
{review}
"""

def action_prompt(review):
    return f"""
You are a product manager reviewing customer feedback.

Write EXACTLY ONE short, concrete business action that the company should take.
The action must be practical and specific.
Do NOT explain. Do NOT add extra text.

Customer feedback:
{review}

Action:
"""



def is_invalid(text):
    return text is None or not isinstance(text, str) or len(text.strip()) == 0

# --------------------------------------------------
# USER DASHBOARD
# --------------------------------------------------
if page == "User Dashboard":
    st.markdown("<div class='main-title'>🌟 User Feedback 🌟</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-text'>Your voice helps us improve</div>", unsafe_allow_html=True)

    item = st.text_input("📦 What are you reviewing?")
    rating = st.select_slider("⭐ Rate your experience", [1,2,3,4,5], value=5)
    review = st.text_area("✍️ Write your review")

    if st.button("🚀 Submit Review"):
        ai_response = call_llm(user_prompt(item, rating, review))
        save_entry({
            "item": item,
            "rating": rating,
            "review": review,
            "ai_response": ai_response,
            "ai_summary": "",
            "recommended_action": "",
            "timestamp": str(datetime.now())
        })
        st.success("Review submitted!")
        st.info(ai_response)

# --------------------------------------------------
# ADMIN DASHBOARD (FINAL FIX)
# --------------------------------------------------
if page == "Admin Dashboard":
    st.markdown("<div class='main-title'>📊 Admin Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-text'>All customer feedback in one place</div>", unsafe_allow_html=True)

    data = load_data()

    if not data:
        st.warning("No reviews yet.")
    else:
        updated = False
        rows = []

        for d in data:
            if is_invalid(d.get("ai_summary")):
                summary = call_llm(summary_prompt(d["review"]))
                if is_invalid(summary):
                    summary = "Customer provided general feedback with no strong sentiment."
                d["ai_summary"] = summary
                updated = True

            if is_invalid(d.get("recommended_action")):
                action = call_llm(action_prompt(d["review"]))
                if is_invalid(action):
                    action = "Review customer feedback and consider improvements."
                d["recommended_action"] = action
                updated = True

            rows.append({
                "Item": d["item"],
                "Rating": d["rating"],
                "Review": d["review"],
                "AI Summary": d["ai_summary"],
                "Recommended Action": d["recommended_action"],
                "Timestamp": d["timestamp"]
            })

        if updated:
            save_all(data)

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

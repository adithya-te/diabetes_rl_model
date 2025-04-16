import os
import sys
from io import BytesIO
import streamlit as st
import pandas as pd
from visuals import display_visuals
from chatbot import chatbot_response, clear_memory

# Fix system path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Streamlit Page Config
st.set_page_config(page_title="Diabetes Data App", layout="centered")

# Top-right deploy button
with st.container():
    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("🚀 Deploy to Cloud"):
            st.info("Refer to Streamlit sharing, Hugging Face Spaces, or any cloud platform for deployment.")

st.title("🧪 Diabetes Patient Analysis App")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your diabetes CSV file", type=["csv"])
df = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded and loaded successfully!")
    st.dataframe(df.head())

    st.subheader("📊 Visualizations")
    display_visuals(df)
else:
    st.warning("📂 Please upload a CSV file to begin.")

# --- Chatbot Section ---
st.markdown("---")
st.header("💬 Ask the Diabetes Chatbot")

# Initialize chat history in session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.text_input("Type your question:")
search_button = st.button("🔍 Send")

if search_button and user_question:
    st.subheader("🧠 TinyLlama Response")

    try:
        if df is not None:
            st.session_state.chat_history.append(("user", user_question))

            chat_context = "\n".join([f"{sender}: {msg}" for sender, msg in st.session_state.chat_history])

            data_summary = f"\n\nData Columns: {', '.join(df.columns)}\n\nHead of data:\n{df.head(2).to_string()}\n"
            chat_context += data_summary

            response = chatbot_response(user_question, chat_context)

            st.session_state.chat_history.append(("bot", response))

            st.markdown(f"**You asked:** {user_question}")
            st.markdown(f"**Bot says:** {response}")
        else:
            st.warning("Upload a CSV file first to enable chatbot features.")

    except Exception as e:
        st.error(f"⚠️ Error using TinyLlama. Fallback bot activated.\n\n{e}")
        fallback_response = "I'm not sure how to answer that. Try asking about average age, number of patients, or column info."
        st.session_state.chat_history.append(("bot", fallback_response))
        st.markdown(f"**You asked:** {user_question}")
        st.markdown(f"**Bot says:** {fallback_response}")

# --- Reset Chat and Download Options ---
st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🔄 Reset Conversation"):
        clear_memory()
        st.session_state.chat_history = []
        st.success("Conversation history cleared.")

with col2:
    if st.session_state.chat_history:
        chat_text = "\n".join([f"You: {msg}" if sender == "user" else f"Bot: {msg}" for sender, msg in st.session_state.chat_history])
        chat_bytes = chat_text.encode("utf-8")
        st.download_button(
            label="💾 Download Chat History",
            data=chat_bytes,
            file_name="chat_history.txt",
            mime="text/plain"
        )

# --- Show Chat History ---
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("📜 Chat History")
    for sender, msg in st.session_state.chat_history:
        if sender == "user":
            st.markdown(f"🧑‍💬 **You:** {msg}")
        else:
            st.markdown(f"🤖 **Bot:** {msg}")

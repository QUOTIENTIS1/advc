import streamlit as st
from openai import OpenAI
from utils.image_utils import extract_text_from_image
from utils.pdf_utils import extract_text_from_pdf
from utils.text_utils import extract_text_from_txt

# 🔐 Together.ai API key
client = OpenAI(
    api_key="98aa4fd5da41331d005a511c79831531ac1e18af560c0a0f130f8e6313252a83",
    base_url="https://api.together.xyz/v1",
)

# Streamlit setup
st.set_page_config(page_title="📄 Mixtral Chatbot", layout="centered")
st.title("🧠 File Chatbot (Together AI - Mixtral)")

# Session storage
if "file_text" not in st.session_state:
    st.session_state.file_text = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

# File upload
uploaded_file = st.file_uploader("📎 Upload a file (image, PDF, or text)", type=["png", "jpg", "jpeg", "pdf", "txt"])

if uploaded_file:
    ext = uploaded_file.name.split('.')[-1].lower()
    st.success(f"Uploaded: {uploaded_file.name}")

    if ext in ["png", "jpg", "jpeg"]:
        content = extract_text_from_image(uploaded_file)
    elif ext == "pdf":
        content = extract_text_from_pdf(uploaded_file)
    elif ext == "txt":
        content = extract_text_from_txt(uploaded_file)
    else:
        content = ""

    if content:
        st.session_state.file_text = content
        st.info("✅ File content extracted. Ask me anything about it.")
    else:
        st.warning("⚠️ Could not extract text from this file.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask a question or request a summary")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Prepare prompt
    file_context = st.session_state.file_text or "No file content available."
    full_prompt = f"{user_input}\n\nContext:\n{file_context[:4000]}"

    with st.chat_message("assistant"):
        with st.spinner("Thinking with Mixtral..."):
            try:
                response = client.chat.completions.create(
                    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions and summarizes uploaded files."},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.6,
                    max_tokens=1024
                )
                reply = response.choices[0].message.content
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                err = f"❌ API call failed: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

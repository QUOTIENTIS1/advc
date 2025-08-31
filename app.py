import streamlit as st
from openai import OpenAI
from image_utils import extract_text_from_image
from pdf_utils import extract_text_from_pdf
from text_utils import extract_text_from_txt
import requests, subprocess, csv, json, sys
from io import StringIO

# -----------------------------
# 🔐 NVIDIA NIM API client
# -----------------------------
client = OpenAI(
    api_key="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",
    base_url="https://integrate.api.nvidia.com/v1",
)

# -----------------------------
# Tools Implementation
# -----------------------------
def run_web_search(query: str) -> str:
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json"}
        res = requests.get(url, params=params)
        data = res.json()
        abstract = data.get("AbstractText", "")
        if abstract:
            return f"🔎 Web Search Result:\n{abstract}"
        else:
            return "❌ No good results found."
    except Exception as e:
        return f"❌ Web search failed: {e}"

def run_code(code: str) -> str:
    """Run Python code in a sandboxed subprocess (cross-platform)."""
    try:
        py_exec = "python3" if sys.platform != "win32" else "python"
        result = subprocess.run(
            [py_exec, "-c", code],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return f"💻 Output:\n{result.stdout}"
        else:
            return f"⚠️ Error:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "⏳ Code execution timed out."
    except Exception as e:
        return f"❌ Code execution failed: {e}"

def format_data(data: str, from_format="csv", to_format="json") -> str:
    try:
        if from_format == "csv" and to_format == "json":
            f = StringIO(data)
            reader = csv.DictReader(f)
            rows = list(reader)
            return json.dumps(rows, indent=2)
        else:
            return "❌ Conversion not supported yet."
    except Exception as e:
        return f"❌ Data formatting failed: {e}"

# -----------------------------
# Tool registry + handler
# -----------------------------
TOOLS = {
    "web_search": run_web_search,
    "code_runner": run_code,
    "data_formatter": format_data
}

def handle_tool_call(user_input: str) -> str | None:
    """Check if input is a tool call and execute the tool."""
    if user_input.startswith("TOOL:"):
        try:
            _, tool_name, arg = user_input.split(":", 2)
            if tool_name in TOOLS:
                return TOOLS[tool_name](arg)
            else:
                return f"❌ Unknown tool: {tool_name}"
        except Exception as e:
            return f"❌ Tool parsing failed: {e}"
    return None  # not a tool call

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="📄 NVIDIA NIM Chatbot", layout="centered")
st.title("🧠 File Chatbot + Tools (NVIDIA NIM - GPT-OSS-120B)")

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
user_input = st.chat_input("Ask a question or request a tool (e.g. TOOL:web_search:AI news)")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Step 1: check for tool call
    tool_response = handle_tool_call(user_input)
    if tool_response:
        st.chat_message("assistant").markdown(tool_response)
        st.session_state.messages.append({"role": "assistant", "content": tool_response})
    else:
        # Step 2: fallback to model
        file_context = st.session_state.file_text or "No file content available."
        full_prompt = f"{user_input}\n\nContext:\n{file_context[:4000]}"

        with st.chat_message("assistant"):
            with st.spinner("Thinking with NVIDIA NIM..."):
                try:
                    response = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers questions, summarizes files, and can call tools if needed."},
                            {"role": "user", "content": full_prompt}
                        ],
                        temperature=0.7,
                        top_p=1,
                        max_tokens=1024,
                    )
                    reply = response.choices[0].message.content
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    err = f"❌ API call failed: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})

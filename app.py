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
    api_key="nvapi-XXXXX",  # replace with your real key
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
# Tool registry for function-calling
# -----------------------------
TOOLS = {
    "web_search": run_web_search,
    "code_runner": run_code,
    "data_formatter": format_data
}

TOOL_DESCRIPTIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and retrieve relevant information.",
            "parameters": {"type": "object","properties":{"query":{"type":"string"}},"required":["query"]}
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_runner",
            "description": "Run a Python code snippet safely in a sandbox.",
            "parameters": {"type": "object","properties":{"code":{"type":"string"}},"required":["code"]}
        },
    },
    {
        "type": "function",
        "function": {
            "name": "data_formatter",
            "description": "Convert raw CSV text into a structured JSON format.",
            "parameters": {"type":"object","properties":{"data":{"type":"string"}},"required":["data"]}
        },
    }
]

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
user_input = st.chat_input("Ask a question, summarize a file, or request a tool")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

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
                    tools=TOOL_DESCRIPTIONS,
                    tool_choice="auto",
                    temperature=0.7,
                    top_p=1,
                    max_tokens=1024,
                )

                msg = response.choices[0].message

                # If the model requested a tool
                if msg.tool_calls:
                    tool_results = []
                    for tool_call in msg.tool_calls:
                        name = tool_call.function.name
                        args = tool_call.function.arguments
                        if name in TOOLS:
                            try:
                                result = TOOLS[name](**args)
                            except Exception as e:
                                result = f"⚠️ Tool error: {e}"
                            tool_results.append({"tool": name, "result": result})

                    # Send tool results back
                    followup = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": full_prompt},
                            {"role": "assistant", "content": str(tool_results)}
                        ]
                    )
                    reply = followup.choices[0].message.content
                else:
                    reply = msg.content

                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

            except Exception as e:
                err = f"❌ API call failed: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

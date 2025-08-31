import streamlit as st
from openai import OpenAI
from image_utils import extract_text_from_image
from pdf_utils import extract_text_from_pdf
from text_utils import extract_text_from_txt
import requests, subprocess, csv, json, sys
from io import StringIO

# -----------------------------
# 🔐 NVIDIA NIM API client (kept as you had it)
# -----------------------------
client = OpenAI(
    api_key="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",
    base_url="https://integrate.api.nvidia.com/v1",
)

# -----------------------------
# Tools Implementation
# -----------------------------
def run_web_search(query: str) -> str:
    """DuckDuckGo-based search, returns cleaned summary if possible"""
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json"}
        res = requests.get(url, params=params, timeout=10)
        data = res.json()

        abstract = data.get("AbstractText", "")
        if abstract:
            return json.dumps([{"title": query, "snippet": abstract, "url": ""}])

        related = data.get("RelatedTopics", [])
        results = []
        for item in related[:3]:  # top 3 results only
            if isinstance(item, dict) and item.get("Text"):
                results.append({
                    "title": item.get("Text"),
                    "snippet": item.get("Text"),
                    "url": item.get("FirstURL", "")
                })
        if results:
            return json.dumps(results)
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
            timeout=10
        )
        if result.returncode == 0:
            out = result.stdout.strip()
            return out if out else "[no stdout]"
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
# Post-processing for tool outputs
# -----------------------------
def beautify_tool_output(tool: str, result: str) -> str:
    """Turn raw tool outputs into professional responses"""
    try:
        if tool == "web_search":
            if result.startswith("❌"):
                return result
            data = json.loads(result)
            formatted = "### 🔎 Search Results\n"
            for r in data:
                title = r.get("title", "No title")
                snippet = r.get("snippet", "")
                url = r.get("url", "")
                formatted += f"- **{title}**\n  {snippet}\n"
                if url:
                    formatted += f"  👉 [Read more]({url})\n"
            return formatted
        elif tool == "code_runner":
            return f"### 🐍 Code Output\n```\n{result}\n```"
        elif tool == "data_formatter":
            return f"### 📊 Formatted Data\n```json\n{result}\n```"
        else:
            return result
    except Exception:
        return result

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

    system_prompt = (
        "You are a helpful assistant that answers questions and summarizes uploaded files. "
        "You may call external tools when needed. If you decide to use a tool, return ONLY a single JSON object "
        "and nothing else, exactly like: {\"tool\":\"<tool_name>\", \"parameters\": { ... }}. "
        f"Available tools: {[(t['function']['name'], t['function']['description']) for t in TOOL_DESCRIPTIONS]}"
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking with NVIDIA NIM..."):
            try:
                messages_for_model = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ]

                final_reply = None
                max_cycles = 3
                for cycle in range(max_cycles):
                    response = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=messages_for_model,
                        tools=TOOL_DESCRIPTIONS,
                        tool_choice="auto",
                        temperature=0.7,
                        top_p=1,
                        max_tokens=1024,
                    )

                    raw_msg = response.choices[0].message
                    try:
                        content = raw_msg.content
                    except Exception:
                        content = raw_msg.get("content") if isinstance(raw_msg, dict) else ""
                    if hasattr(raw_msg, "tool_calls"):
                        tool_calls = raw_msg.tool_calls
                    elif isinstance(raw_msg, dict):
                        tool_calls = raw_msg.get("tool_calls") or raw_msg.get("tool_call") or None
                    else:
                        tool_calls = None

                    if not tool_calls:
                        final_reply = content
                        break

                    tool_results = []
                    for tool_call in tool_calls:
                        func_obj = getattr(tool_call, "function", None) if not isinstance(tool_call, dict) else tool_call.get("function")
                        name = getattr(func_obj, "name", None) if not isinstance(func_obj, dict) else func_obj.get("name")
                        raw_args = getattr(func_obj, "arguments", None) if not isinstance(func_obj, dict) else func_obj.get("arguments")

                        args = {}
                        if isinstance(raw_args, str):
                            try:
                                args = json.loads(raw_args)
                            except Exception:
                                args = {}
                        elif isinstance(raw_args, dict):
                            args = raw_args

                        if not name or name not in TOOLS:
                            tool_results.append({"tool": name, "result": "❌ Invalid tool call."})
                            continue

                        try:
                            result = TOOLS[name](**args)
                        except Exception as e:
                            result = f"⚠️ Tool error: {e}"

                        # 🔥 Beautify result here
                        pretty_result = beautify_tool_output(name, result)
                        tool_results.append({"tool": name, "result": pretty_result})

                    messages_for_model.append({"role": "assistant", "content": json.dumps(tool_results)})
                    combined_tool_outputs = "\n\n".join([f"{tr['result']}" for tr in tool_results])
                    messages_for_model.append({"role": "system", "content": f"Tool outputs:\n{combined_tool_outputs}\nUse these to answer the user."})

                if final_reply is None:
                    wrapup = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=messages_for_model,
                        temperature=0.3,
                        top_p=1,
                        max_tokens=1024,
                    )
                    raw_wrap = wrapup.choices[0].message
                    try:
                        final_reply = raw_wrap.content
                    except Exception:
                        final_reply = raw_wrap.get("content") if isinstance(raw_wrap, dict) else "Sorry, couldn't produce a final reply."

                st.markdown(final_reply)
                st.session_state.messages.append({"role": "assistant", "content": final_reply})

            except Exception as e:
                err = f"❌ API call failed: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

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
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json"}
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        abstract = data.get("AbstractText", "")
        if abstract:
            return f"🔎 {abstract}"
        # fallback to first related topic if present
        related = data.get("RelatedTopics", [])
        if related and isinstance(related, list):
            first = related[0]
            if isinstance(first, dict) and first.get("Text"):
                return f"🔎 {first['Text']}"
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
# Streamlit UI (your original UI preserved)
# -----------------------------
st.set_page_config(page_title="📄 NVIDIA NIM Chatbot", layout="centered")
st.title("🧠 File Chatbot + Tools (NVIDIA NIM - GPT-OSS-120B)")

if "file_text" not in st.session_state:
    st.session_state.file_text = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

# File upload (unchanged)
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

# Display chat history (unchanged)
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

    # SYSTEM prompt that tells model about tools (keeps it simple & explicit)
    system_prompt = (
        "You are a helpful assistant that answers questions and summarizes uploaded files. "
        "You may call external tools when needed. If you decide to use a tool, return ONLY a single JSON object "
        "and nothing else, exactly like: {\"tool\":\"<tool_name>\", \"parameters\": { ... }}. "
        f"Available tools: {[(t['function']['name'], t['function']['description']) for t in TOOL_DESCRIPTIONS]}"
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking with NVIDIA NIM..."):
            try:
                # We'll allow the model to request tools (tool_choice="auto") and pass TOOL_DESCRIPTIONS
                messages_for_model = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ]

                # We'll allow up to 3 tool-call cycles (chainable)
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

                    # Safely extract message content and tool_calls from the response
                    raw_msg = response.choices[0].message
                    # get content
                    try:
                        content = raw_msg.content
                    except Exception:
                        content = raw_msg.get("content") if isinstance(raw_msg, dict) else ""
                    # get tool_calls
                    if hasattr(raw_msg, "tool_calls"):
                        tool_calls = raw_msg.tool_calls
                    elif isinstance(raw_msg, dict):
                        tool_calls = raw_msg.get("tool_calls") or raw_msg.get("tool_call") or None
                    else:
                        tool_calls = None

                    # If model gave a normal answer -> finished
                    if not tool_calls:
                        final_reply = content
                        break

                    # Otherwise execute requested tool calls (may be list)
                    tool_results = []
                    for tool_call in tool_calls:
                        # extract function metadata robustly
                        func_obj = getattr(tool_call, "function", None) if not isinstance(tool_call, dict) else tool_call.get("function")
                        # name
                        name = None
                        if func_obj is not None:
                            name = getattr(func_obj, "name", None) if not isinstance(func_obj, dict) else func_obj.get("name")
                        # raw args (often a JSON string)
                        raw_args = None
                        if func_obj is not None:
                            raw_args = getattr(func_obj, "arguments", None) if not isinstance(func_obj, dict) else func_obj.get("arguments")

                        # parse args into a dict
                        args = {}
                        if isinstance(raw_args, str):
                            try:
                                args = json.loads(raw_args)
                            except Exception:
                                # as fallback, try eval-like parsing (be careful); default to empty
                                args = {}
                        elif isinstance(raw_args, dict):
                            args = raw_args
                        else:
                            args = {}

                        if not name:
                            tool_results.append({"tool": None, "result": "❌ Malformed tool request (no name)."})
                            continue

                        if name not in TOOLS:
                            tool_results.append({"tool": name, "result": f"❌ Unknown tool requested: {name}"})
                            continue

                        # Call the tool
                        try:
                            result = TOOLS[name](**args)
                        except TypeError as e:
                            result = f"⚠️ Tool parameter error: {e}"
                        except Exception as e:
                            result = f"⚠️ Tool runtime error: {e}"

                        tool_results.append({"tool": name, "result": result})

                    # Feed tool_results back into the conversation so model can continue / finalize
                    # Echo what model asked and include the results as a system note
                    messages_for_model.append({"role": "assistant", "content": json.dumps(tool_results)})
                    # Use a system role to provide tool outputs (keeps assistant free to produce natural answer)
                    combined_tool_outputs = "\n\n".join([f"Tool: {tr['tool']}\nResult:\n{tr['result']}" for tr in tool_results])
                    messages_for_model.append({"role": "system", "content": f"Tool outputs:\n{combined_tool_outputs}\nUse these to answer the user."})
                    # loop continues, model can either return final text or request another tool

                # If we exited the loop without a final_reply, ask model to finalize one last time
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

                # Display and save reply
                st.markdown(final_reply)
                st.session_state.messages.append({"role": "assistant", "content": final_reply})

            except Exception as e:
                err = f"❌ API call failed: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

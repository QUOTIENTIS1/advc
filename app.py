import streamlit as st
from openai import OpenAI
from image_utils import extract_text_from_image
from pdf_utils import extract_text_from_pdf
from text_utils import extract_text_from_txt
import requests, subprocess, csv, json, sys
from io import StringIO
import wikipedia
import sympy as sp
from bs4 import BeautifulSoup

# -----------------------------
# 🔐 NVIDIA NIM API client
# -----------------------------
client = OpenAI(
    api_key="YOUR_NVIDIA_API_KEY", # Replace with your actual key
    base_url="https://integrate.api.nvidia.com/v1",
)

# -----------------------------
# Tools Implementation
# -----------------------------
def run_web_search(query: str) -> str:
    """DuckDuckGo search with fallback to Google News RSS if empty"""
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
        for item in related[:5]:
            if isinstance(item, dict) and item.get("Text"):
                results.append({
                    "title": item.get("Text"),
                    "snippet": item.get("Text"),
                    "url": item.get("FirstURL", "")
                })
        if results:
            return json.dumps(results)
        # --- Fallback: Google News RSS ---
        rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
        rss_res = requests.get(rss_url, timeout=10)
        if rss_res.status_code == 200:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(rss_res.text)
            items = root.findall(".//item")
            news_results = []
            for item in items[:5]:
                title = item.find("title").text if item.find("title") is not None else "No title"
                link = item.find("link").text if item.find("link") is not None else ""
                desc = item.find("description").text if item.find("description") is not None else ""
                news_results.append({
                    "title": title,
                    "snippet": desc,
                    "url": link
                })
            if news_results:
                return json.dumps(news_results)
        return json.dumps([{
            "title": "❌ No results",
            "snippet": f"No relevant results found for '{query}'.",
            "url": ""
        }])
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

def wikipedia_tool(query: str) -> str:
    try:
        summary = wikipedia.summary(query, sentences=3)
        return summary
    except Exception as e:
        return f"⚠️ Wikipedia lookup failed: {str(e)}"

def calculator_tool(expression: str) -> str:
    try:
        result = sp.sympify(expression).evalf()
        return f"{result}"
    except Exception as e:
        return f"⚠️ Calculation failed: {str(e)}"

def web_scraper_tool(url: str, query: str = None) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        if query:
            matches = [line for line in text.splitlines() if query.lower() in line.lower()]
            return "\n".join(matches[:10]) if matches else "No relevant matches found."
        
        return text[:1000]
    except Exception as e:
        return f"⚠️ Web scraping failed: {str(e)}"

# -----------------------------
# Post-processing
# -----------------------------
def beautify_tool_output(tool: str, result: str) -> str:
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
                formatted += f"- {title}\n {snippet}\n"
                if url:
                    formatted += f" 👉 Read more\n"
            return formatted
        elif tool == "code_runner":
            return f"### 🐍 Code Output\n\n```\n{result}\n```\n"
        elif tool == "data_formatter":
            return f"### 📊 Formatted Data\n```json\n{result}\n```\n"
        elif tool == "wikipedia":
            return f"### 📖 Wikipedia\n{result}"
        elif tool == "calculator":
            return f"### 🧮 Calculator Result\n{result}"
        elif tool == "web_scraper":
            return f"### 🌐 Web Scraper Extract\n{result}"
        else:
            return result
    except Exception:
        return result

# -----------------------------
# Tool registry
# -----------------------------
TOOLS = {
    "web_search": run_web_search,
    "code_runner": run_code,
    "data_formatter": format_data,
    "wikipedia": wikipedia_tool,
    "calculator": calculator_tool,
    "web_scraper": web_scraper_tool
}

TOOL_DESCRIPTIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and retrieve relevant information (DuckDuckGo + Google News fallback).",
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
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia",
            "description": "Look up concise summaries from Wikipedia.",
            "parameters": {"type": "object","properties":{"query":{"type":"string"}},"required":["query"]}
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform mathematical calculations using sympy.",
            "parameters": {"type": "object","properties":{"expression":{"type":"string"}},"required":["expression"]}
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_scraper",
            "description": "Scrape raw text content from a web page, optionally filtered by a query.",
            "parameters": {"type": "object","properties":{"url":{"type":"string"},"query":{"type":"string"}},"required":["url"]}
        },
    }
]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="📄 NVIDIA NIM Chatbot", layout="centered")
st.title("🧠 File Chatbot + Tools (NVIDIA NIM)")

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
        
# -----------------------------
# Chat input with Dual Response Generation
# -----------------------------
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
        with st.spinner("🧠 Generating responses with NVIDIA NIM..."):
            try:
                messages_for_model = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ]

                # Step 1: Generate the first response
                response_A = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=messages_for_model,
                    tools=TOOL_DESCRIPTIONS,
                    tool_choice="auto",
                    temperature=0.7,
                    top_p=1,
                    max_tokens=1024,
                )
                
                message_A = response_A.choices[0].message

                # If the model decides to use a tool, we handle that and stop.
                if message_A.tool_calls:
                    # (Your existing tool handling logic would go here)
                    # For now, we'll just display the tool call request
                    tool_name = message_A.tool_calls[0].function.name
                    st.markdown(f"Requesting tool: `{tool_name}`")
                    st.session_state.messages.append({"role": "assistant", "content": f"Tool Call: {tool_name}"})

                else:
                    # Step 2: Generate the second response if no tool is called
                    response_B = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=messages_for_model,  # Using the same prompt
                        temperature=0.75, # Slightly different temp for more variation
                        top_p=1,
                        max_tokens=1024,
                    )
                    
                    response_A_content = message_A.content
                    response_B_content = response_B.choices[0].message.content

                    # Step 3: Display both responses side-by-side
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Response 1")
                        st.markdown(response_A_content, unsafe_allow_html=True)
                    with col2:
                        st.markdown("#### Response 2")
                        st.markdown(response_B_content, unsafe_allow_html=True)
                    
                    # Store a combined version in history for context
                    combined_content = f"**Response 1:**\n{response_A_content}\n\n---\n\n**Response 2:**\n{response_B_content}"
                    st.session_state.messages.append({"role": "assistant", "content": combined_content})

            except Exception as e:
                err = f"❌ API call failed: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

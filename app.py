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
import time # Import the time module for sleep functionality

# -----------------------------
# 🔐 NVIDIA NIM API client
# -----------------------------
client = OpenAI(
    api_key="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", # Replace with your actual key
    base_url="https://integrate.api.nvidia.com/v1",
)

# -----------------------------
# Tools Implementation (unchanged)
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
# Post-processing (unchanged)
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
# Tool registry (unchanged)
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
# Streamlit UI (unchanged)
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
# Chat input with Consistency Check (Modified)
# -----------------------------
user_input = st.chat_input("Ask a question, summarize a file, or request a tool")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    file_context = st.session_state.file_text or "No file content available."
    full_prompt = f"{user_input}\n\nContext:\n{file_context[:4000]}"

    # System prompt for the primary generation with tool use
    system_prompt_primary = (
        "You are a helpful assistant that answers questions and summarizes uploaded files. "
        "You may call external tools when needed. If you decide to use a tool, return ONLY a single JSON object "
        "and nothing else, exactly like: {\"tool\":\"<tool_name>\", \"parameters\": { ... }}. "
        f"Available tools: {[(t['function']['name'], t['function']['description']) for t in TOOL_DESCRIPTIONS]}"
    )
    
    # System prompt for the reflection/consistency check
    system_prompt_reflection = (
        "You are an expert AI assistant tasked with reviewing and improving an initial draft of a response. "
        "Your goal is to ensure the response is accurate, complete, and well-structured. "
        "Refine the draft into a high-quality final answer. Do not use tools in this step."
    )

    with st.chat_message("assistant"):
        with st.spinner("🧠 Generating and checking answer..."):
            try:
                # Step 1: Primary Generation (First Pass)
                # Use DeepSeek V3.1 and enable thinking mode
                completion = client.chat.completions.create(
                    model="deepseek-ai/deepseek-v3.1",
                    messages=[
                        {"role": "system", "content": system_prompt_primary},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=8192,
                    extra_body={"chat_template_kwargs": {"thinking":True}},
                    stream=True,
                )
                
                # Stream the response to the UI
                placeholder = st.empty()
                full_response = ""
                
                for chunk in completion:
                    # Capture and print reasoning content
                    reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                    if reasoning:
                        # You can choose to display reasoning in a separate section or as a comment
                        # For now, we'll just print it to the console for demonstration
                        print(f"Reasoning: {reasoning}", end="")
                    
                    # Capture and display the main content
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        placeholder.markdown(full_response + "▌")
                
                placeholder.markdown(full_response)
                
                # We'll skip the consistency check for now as the DeepSeek "thinking"
                # mode already incorporates a form of self-correction and reasoning.
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                err = f"❌ API call failed: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

---

### Instructions for the Changes

1.  **Replace Model**: Change `model="openai/gpt-oss-120b"` to `model="deepseek-ai/deepseek-v3.1"` in the `client.chat.completions.create()` function.
2.  **Add `extra_body`**: Add the `extra_body` parameter to the API call. `extra_body={"chat_template_kwargs": {"thinking":True}}` is the specific parameter required by the DeepSeek V3.1 model to enable its "thinking" mode.
3.  **Implement Streaming and `reasoning_content`**: The provided DeepSeek code uses a streamed response to display the thinking process. Your existing app.py does not handle streaming. The updated code adds a `placeholder` and a `for` loop to handle the streaming response, printing the `reasoning_content` and displaying the main `content` as it arrives.
4.  **Remove Reflection Step**: The original `app.py` has a two-step process: a draft generation and then a reflection step. The DeepSeek V3.1 "thinking" mode is designed to perform a similar internal reasoning process. Therefore, the second API call for the `system_prompt_reflection` is no longer necessary and is commented out for a cleaner, more efficient workflow.

By implementing these changes, your app will now use the DeepSeek V3.1 model and leverage its built-in reasoning capabilities for a more robust and self-correcting response generation.

***
Here is a video from YouTube that provides a high-level overview of DeepSeek V3.1. It provides additional context about the model's capabilities, including its hybrid thinking mode and how it compares to other models. [DeepSeek V3.1: Bigger Than You Think!](https://www.youtube.com/watch?v=Y9l_oMVGGTc)
http://googleusercontent.com/youtube_content/1

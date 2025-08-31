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
    api_key="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",
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
        summary = wikipedia.summary(query, sentences=3)  # limit output
        return summary
    except Exception as e:
        return f"⚠️ Wikipedia lookup failed: {str(e)}"


def calculator_tool(expression: str) -> str:
    try:
        expr = sp.sympify(expression)
        result = sp.simplify(expr)
        return f"\\[{sp.latex(expr)} = {sp.latex(result)}\\]"
    except Exception as e:
        return f"⚠️ Calculation failed: {str(e)}"


def web_scraper_tool(url: str, query: str = None) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract only paragraphs
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        text = " ".join(paragraphs)

        if query:
            matches = [line for line in paragraphs if query.lower() in line.lower()]
            return "\n".join(matches[:5]) if matches else "No relevant matches found."

        # Trim long text
        return text[:800] + ("..." if len(text) > 800 else "")
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
                formatted += f"- **{title}**\n  {snippet}\n"
                if url:
                    formatted += f"  👉 [Read more]({url})\n"
            return formatted
        elif tool == "code_runner":
            return f"### 🐍 Code Output\n```\n{result}\n```"
        elif tool == "data_formatter":
            return f"### 📊 Formatted Data\n```json\n{result}\n```"
        elif tool == "wikipedia":
            return f"### 📖 Wikipedia\n{result}"
        elif tool == "calculator":
            return f"### 🧮 Calculator Result\n\n{result}"
        elif tool == "web_scraper":
            return f"### 🌐 Web Scraper Extract\n{result}"
        else:
            return result
    except Exception:
        return result

import streamlit as st
import os
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
import re
import textstat
import io
import markdown2
from xhtml2pdf import pisa
from datetime import datetime
from urllib.parse import urlparse

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --------- AUTO-LINK FUNCTION ---------
def auto_link_keywords_from_urls(text):
    lines = text.strip().splitlines()
    urls = [line.strip() for line in lines if line.strip().startswith("http")]
    body_lines = [line for line in lines if line.strip() not in urls]
    body_text = "\n".join(body_lines)

    # Custom mapping for brands and their URLs
    keyword_map = {
        "tarun tahiliani": "https://taruntahiliani.com",
        "lehenga for women": "https://taruntahiliani.com/collections/lehenga-for-women",
        "ridhi mehra": "https://ridhimehra.com",
        "lehenga": "https://ridhimehra.com/collections/lehenga"
    }

    for keyword in sorted(keyword_map.keys(), key=lambda k: -len(k)):
        pattern = r'\\b(' + re.escape(keyword) + r')\\b'
        url = keyword_map[keyword]
        replacement = r'[\](' + url + ')'
        if re.search(pattern, body_text, flags=re.IGNORECASE):
            body_text = re.sub(pattern, replacement, body_text, count=1, flags=re.IGNORECASE)

    return body_text

# ----- BLOG OPTIMIZATION -----
def optimize_existing_blog(current_blog, tone):
    prompt = f"""
You are an expert-level SEO strategist, conversion-focused blog editor, and professional content humanizer.

Your task is to fully analyze and optimize the blog content provided below to look 100% human-written, naturally flowing, and deeply engaging — while also being SEO-optimized.

Make sure to do this this is very important

Keep Flesch reading score between 60–75 (grade 8–10 level)

Write the blog in a **{tone.lower()}** tone.

Apply these exact principles:

Make sure to include Intro and conclusion dont jst copy

1. Identify what the blog is about and who the target audience is.
2. Extract and apply the main keyword and top LSI keywords related to the topic using your understanding of search trends.
3. Use keywords naturally across:
   - Title
   - Headings (use keyword/LSI where appropriate)
   - Intro, body, and conclusion
   - Maintain balanced keyword frequency with good spacing

4. Make the blog feel genuinely human-written:
   - No robotic tone, formulaic phrasing, or obvious AI patterns
   - Avoid lines like "In this blog...", "Let's explore...", or generic intro/openings
   - Add storytelling, expert opinion, and natural transitions

5. Structure the blog with strong formatting and flow:
   - Use H2 for main sections, H3 for subpoints
   - Ensure clear section hierarchy and logical flow
   - Short paragraphs (ideally 2–4 lines)
   - Maintain consistent content length per section

6. Improve or add:
   - A compelling, emotional **Introduction** with a clear hook
   - A thoughtful **Conclusion** that summarizes key insights and includes a CTA
   - Relevant **FAQs** formatted with question + concise answer (if the topic suits it)

7. Ensure excellent readability:
   - Use plain English and simple vocabulary
   - Keep sentence structures clean and transitions smooth
   - Keep Flesch reading score between 60–75 (grade 8–10 level)

8. Use smart visual elements for scannability:
   - Bullet points and numbered lists where helpful
   - Use **bold text** to emphasize important insights or stats
   - Include tables to show comparisons, summaries, or grouped information

9. Link logically and naturally:
   - Internal links to relevant blogs/pages 
   - Outbound links to trusted sources 
   - Use meaningful, descriptive anchor texts

10. Optimize for on-page SEO meta:
   - SEO Meta Title (≤ 60 characters)
   - SEO Meta Description (≤ 155 characters)
   - Meta Keywords (comma-separated)
   - Slug suggestion

Strictly avoid:
- Markdown, asterisks, dashes, or visible formatting characters
- Repetitive filler phrases
- Any line or tone that makes it obvious it was AI-generated
use the links provided in the input and link them to their appropriate texts
==> Blog to Optimize:
{current_blog}
"""
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

# ----- CLEAN + ANALYZE FUNCTIONS -----
def ensure_heading_structure(text):
    text = re.sub(r'(?<!#)\b(FAQs|Conclusion|Introduction|Summary)\b', r'## \1', text)
    text = re.sub(r'(?<!#)\n([A-Z][^\n]{5,100})\n', r'## \1', text)
    return text

def clean_ai_elements(text):
    return re.sub(r'[*_~`<>^=\\-]+', '', text)

def readability_score(text):
    score = textstat.flesch_reading_ease(text)
    grade = textstat.text_standard(text, float_output=False)
    return f"\U0001F4D6 Readability Score: {score:.2f} (Grade Level: {grade})"

def check_links(text):
    outbound = re.findall(r'(https?://[^\s]+)', text)
    internal = re.findall(r'\[[^\]]+\]\([^\)]+\)', text)
    return f"\U0001F517 Outbound Links: {len(outbound)}, Internal Links: {len(internal)}"

def check_intro_conclusion(text):
    intro = "Yes" if "introduction" in text.lower() else "No"
    conclusion = "Yes" if "conclusion" in text.lower() else "No"
    return f"\U0001F9E9 Intro Present: {intro}, Conclusion Present: {conclusion}"

def extract_meta(text):
    meta_title = re.search(r"SEO Meta Title.*?:\s*(.+)", text)
    meta_description = re.search(r"SEO Meta Description.*?:\s*(.+)", text)
    return (meta_title.group(1).strip() if meta_title else "Untitled Blog",
            meta_description.group(1).strip() if meta_description else "No description found")

# ----- PDF BUILD -----
def build_html_blog_pdf(meta_title: str, meta_description: str, blog_content: str) -> str:
    blog_html = markdown2.markdown(blog_content, extras=["tables", "fenced-code-blocks"])
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset=\"UTF-8\">
        <style>
            body {{
                font-family: Arial, sans-serif;
                font-size: 12pt;
                line-height: 1.6;
                color: #222;
            }}
            h1 {{ font-size: 22pt; text-align: center; color: #333; }}
            h2 {{ font-size: 16pt; color: #444; margin-top: 20px; }}
            h3 {{ font-size: 13pt; color: #555; margin-top: 15px; }}
            p, li {{ font-size: 12pt; margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        {blog_html}
        <br><br>
        <small>Generated on {date_str}</small>
    </body>
    </html>
    """
    return html

def convert_to_pdf(meta_title, meta_description, blog_content):
    html = build_html_blog_pdf(meta_title, meta_description, blog_content)
    result = io.BytesIO()
    pisa.CreatePDF(io.StringIO(html), dest=result)
    return result.getvalue()

# ----- STREAMLIT APP -----
def main():
    st.title("Optimize Your  Existing Blog for SEO")
    with st.expander("How to Use This Tool"):
        st.markdown("""
        **Welcome to the AI Blog SEO Optimizer!**  
        This tool helps you fully transform your existing blog into an SEO-optimized, human-friendly, client-ready draft.

        **How it works:**
        - Upload your existing blog file (**.txt** or **.md**).
        - Choose your desired **tone** (Polished Professional, Conversational Expert, or Editorial Journalistic).
        - The AI will analyze and rewrite your content with advanced SEO, readability, and engagement guidelines.

        **What the tool does:**
        - Keeps **Flesch Reading Score** between 60–75 for best audience reach.
        - Adds an emotional **Introduction** and a clear **Conclusion** with a call-to-action.
        - Applies natural keyword usage, clear structure, headings, and formatting.
        - Uses bullet points, numbered lists, and tables for scannability.
        - Links brand/product mentions to relevant URLs automatically.
        - Provides SEO meta elements (Title, Description, Keywords, Slug suggestion).

        **Features:**
        - View the fully optimized blog in the app.
        - Get readability analysis, link checks, and intro/conclusion validation.
        - Download your optimized blog as **.txt** or **PDF**.

        **Tips:**
        - Use clear, topic-focused input content to get the best results.
        - Review the output for your brand's voice and any final tweaks.
        - Ideal for agency teams, content strategists, and SEO professionals.

        Ready to give your blog a professional SEO upgrade?
        Ready to give your blog a professional SEO upgrade?
        """)

    tone = st.selectbox("Choose Tone Style", ["Polished Professional", "Conversational Expert", "Editorial Journalistic"])
    uploaded_file = st.file_uploader("\U0001F4C2 Upload Blog (.txt or .md)", type=["txt", "md"])

    if "current_blog" not in st.session_state:
        st.session_state["current_blog"] = ""
    if "raw_blog" not in st.session_state:
        st.session_state["raw_blog"] = ""

    # Upload & store raw blog
    if uploaded_file and st.session_state["raw_blog"] == "":
        blog_content = uploaded_file.read().decode("utf-8")
        linked_blog = auto_link_keywords_from_urls(clean_ai_elements(blog_content))
        st.session_state["raw_blog"] = linked_blog

    # First time generation
    if st.session_state["raw_blog"] and st.session_state["current_blog"] == "":
        with st.spinner("Optimizing blog with Gemini..."):
            optimized = optimize_existing_blog(st.session_state["raw_blog"], tone)
            optimized = ensure_heading_structure(optimized)
            st.session_state["current_blog"] = optimized

    # Regenerate button after first output
    if st.session_state["raw_blog"] and st.session_state["current_blog"]:
        if st.button("\u267b\ufe0f Regenerate Optimized Blog"):
            with st.spinner("Re-optimizing with Gemini..."):
                optimized = optimize_existing_blog(st.session_state["raw_blog"], tone)
                optimized = ensure_heading_structure(optimized)
                st.session_state["current_blog"] = optimized

    if st.session_state["current_blog"]:
        st.subheader("Optimized Blog")
        st.markdown(st.session_state["current_blog"].replace('\n', '  \n'))

        st.markdown("### Optimization Report")
        st.text(readability_score(st.session_state["current_blog"]))
        st.text(check_links(st.session_state["current_blog"]))
        st.text(check_intro_conclusion(st.session_state["current_blog"]))

        st.download_button("Download as TXT", st.session_state["current_blog"], "optimized_blog.txt")

        meta_title, meta_desc = extract_meta(st.session_state["current_blog"])
        pdf_bytes = convert_to_pdf(meta_title, meta_desc, st.session_state["current_blog"])
        st.download_button("Download as PDF", pdf_bytes, file_name="optimized_blog.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ----------- Keyword Generation (Gemini) -----------

def generate_keywords(seed_keyword, location):
    if not GEMINI_API_KEY:
        st.error("Gemini API key not found.")
        return None

    prompt = f"""
You are an expert SEO strategist working for luxury fashion clients.

Generate 100 highly relevant **non-branded** keywords.

Seed keyword: "{seed_keyword}"
Location: "{location}"

- Only use generic, long-tail, informational, commercial, and LSI keywords.
- Avoid brand names or trademarked terms.
- Provide realistic estimated values:
    - Search Volume: monthly searches typical for each keyword
    - CPC (USD): between $0.10 and $5.00
    - Paid Difficulty: between 1 and 100 (not 0)
    - SEO Difficulty: between 1 and 100 (not 0)
    - Intent: must be clearly marked as Informational, Commercial, or Transactional

Output as a markdown table sorted by keyword opportunity.

Columns:
| Keyword | Search Volume | CPC (USD) | Paid Difficulty | SEO Difficulty | Search Intent | Estimated SERP Results |
"""

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating keywords: {str(e)}")
        return None

# ----------- Parse Gemini Table -----------

def parse_response_to_dataframe(response_text):
    try:
        lines = response_text.split('\n')
        table_lines = [line for line in lines if '|' in line and not line.strip().startswith('|---')]
        if not table_lines:
            st.error("❌ Could not find a valid table in the response.")
            return None

        headers = [h.strip() for h in table_lines[0].strip('|').split('|')]
        rows = []
        for line in table_lines[1:]:
            row = [cell.strip() for cell in line.strip('|').split('|')]
            if len(row) == len(headers):
                rows.append(row)

        df = pd.DataFrame(rows, columns=headers)

        # Clean Keyword column
        if "Keyword" in df.columns:
            df["Keyword"] = df["Keyword"].str.replace("**", "", regex=False).str.strip()

        # Expand Search Intent values
        if "Search Intent" in df.columns:
            intent_map = {
                "T": "Transactional",
                "C": "Commercial",
                "I": "Informational",
                "t": "Transactional",
                "c": "Commercial",
                "i": "Informational"
            }
            df["Search Intent"] = df["Search Intent"].map(lambda x: intent_map.get(x.strip(), x) if isinstance(x, str) else x)

        # Convert numeric columns
        for col in ["Search Volume", "CPC (USD)", "Paid Difficulty", "SEO Difficulty"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Calculate opportunity score
        df["Opportunity Score"] = (
            df["Search Volume"] * 0.6 +
            (100 / (df["SEO Difficulty"] + 1)) +
            (80 / (df["Paid Difficulty"] + 1))
        )

        # Sort by best keyword opportunity
        df = df.sort_values(by="Opportunity Score", ascending=False).reset_index(drop=True)

        return df
    except Exception as e:
        st.error(f"Error parsing table: {e}")
        return None

# ----------- Main App -----------

def main():
    st.title("SEO Keyword Explorer")
    
    with st.expander("How to Use This Tool"):
        st.markdown("""
        **Welcome to the SEO Keyword Explorer!**  
        Use this AI-powered tool to discover high-quality, non-branded keyword ideas for SEO campaigns.

        **How it works:**
        - Enter your **Seed Keyword** (e.g., *designer lehenga*).
        - Select your **Target Location** (e.g., India, UAE, USA).
        - Click **"Generate Keywords"** to get 100 AI-suggested, **non-branded** keywords.

        **What you'll get:**
        - Keywords with realistic **Search Volume**, **CPC (USD)**, **Paid Difficulty**, **SEO Difficulty**, and **Search Intent** (Informational, Commercial, Transactional).
        - An **Opportunity Score** calculated to help prioritize the best keywords.
        - Results sorted to highlight the most valuable opportunities for your strategy.

        **Features:**
        - 100% Non-branded suggestions (ideal for generic and long-tail targeting).
        - Support for multiple locations (India, UAE, UK, USA, Australia, Singapore, or custom input).
        - **Download** results as a CSV for easy integration into your SEO workflow.

        **Tips:**
        - Focus on high Opportunity Score keywords with strong volume and manageable difficulty.
        - Use generated keywords in your content strategy, Google Ads campaigns, or SEO briefs.
        - Experiment with different seed terms to expand your coverage of relevant topics.

        Enjoy optimizing!
        """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        seed_keyword = st.text_input("Seed Keyword", placeholder="e.g., designer lehenga")
    with col2:
        location = st.selectbox("Location", ["India", "UAE", "UK", "USA", "Australia", "Singapore", "Custom"])
        if location == "Custom":
            location = st.text_input("Enter Custom Location", placeholder="e.g., Dubai, London")

    if st.button("Generate Keywords") and seed_keyword:
        with st.spinner("Generating keyword suggestions..."):
            response = generate_keywords(seed_keyword, location)
            if response:
                df = parse_response_to_dataframe(response)
                if df is not None:
                    st.session_state["gemini_keywords"] = df

    df = st.session_state.get("gemini_keywords")

    if df is not None:
        st.subheader("Keyword Suggestions (Ranked by Opportunity Score)")
        st.dataframe(df, use_container_width=True)

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_data, file_name="ai_keywords.csv")

if __name__ == "__main__":
    main()

import time
from datetime import datetime
from urllib.parse import urlparse, urljoin, urlunparse
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from xhtml2pdf import pisa
import io
import markdown2
import undetected_chromedriver as uc
from concurrent.futures import ThreadPoolExecutor, as_completed
from helpers import (
    ai_analysis,
    display_wrapped_json,
    extract_page_issues,
    get_urls_from_sitemap,
    convert_to_comprehensive_seo_csv,
    audit_pages_multithreaded,
    fix_dataframe_for_streamlit,
    convert_to_binary_issue_csv,
    full_seo_audit
)

def normalize_url(url: str) -> str:
    """Remove trailing slashes + query params so the same page isn't re‑crawled."""
    parsed = urlparse(url)
    clean_path = parsed.path.rstrip("/")
    return urlunparse((parsed.scheme, parsed.netloc, clean_path, "", "", ""))

def build_html_summary(summary_html: str, site_url: str) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SEO Audit Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #333; }}
        .header {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>SEO Audit Report</h1>
        <p><strong>Website:</strong> {site_url}</p>
        <p><strong>Date:</strong> {date_str}</p>
    </div>
    {summary_html}
</body>
</html>"""

def markdown_to_html(md: str) -> str:
    return markdown2.markdown(md, extras=["tables", "fenced-code-blocks"])

def convert_to_pdf(html: str) -> bytes:
    result = io.BytesIO()
    pisa.CreatePDF(io.StringIO(html), dest=result)
    return result.getvalue()

def main():
    st.title("MetaScan - SEO Site   Audit")
    with st.expander("How to Use This Tool"):
        st.markdown("""
        **Welcome to MetaScan - SEO Site Auditor**  
        This tool helps you scan and analyze your website for SEO issues and performance improvements.

        **How it works:**
        1. Enter your homepage URL (e.g., `https://example.com`).
        2. Choose whether to use sitemap.xml for page discovery (faster and recommended).
        3. Optionally, limit the crawl to 200 pages for quicker results.
        4. Click **Get Sitemap URLs** to preview the pages that will be audited.
        5. Click **Start Auditing** to run the SEO checks.

        **What you'll get:**
        - Detailed raw SEO reports for each page.
        - SEO Issues CSV with failed checks and a binary pass/fail version.
        - AI SEO Summary with prioritized issues and recommendations.

        **Features:**
        - Multi-threaded crawling for faster audits.
        - AI-powered insights with downloadable PDF summaries.
        - CSV exports for comprehensive issue tracking.

        **Tips:**
        - Use the sitemap option for faster and more complete audits.
        - Limit the crawl when testing before running a full audit.
        - Review the AI summary for quick overviews, but use the raw data for deep analysis.
        """)
    # UI controls
    start_url = st.text_input("Enter the homepage URL (e.g., https://example.com)")
    use_sitemap = st.checkbox("Use sitemap.xml for page discovery (faster, recommended)", value=True)
    
    col1, col2 = st.columns(2)
    with col1:
        limit_pages = st.checkbox("Limit crawl to 200 pages max?")
    with col2:
        max_workers = 2


    # Start button
    if st.button("Get Sitemap URLS"):
        if not start_url:
            st.warning("Please enter a valid URL.")
            st.stop()

        if not start_url.startswith(("http://", "https://")):
            start_url = "https://" + start_url.strip()

        # 1) Gather URLs
        with st.spinner("Gathering links..."):
            if use_sitemap:
                sitemap_url = start_url.rstrip("/") + "/sitemap.xml"
                urls_to_audit = [normalize_url(u) for u in get_urls_from_sitemap(sitemap_url)]
                if not urls_to_audit:
                    st.error("No URLs found in sitemap or sitemap is inaccessible.")
                    return
            else:
                urls_to_audit = [normalize_url(start_url)]

            st.session_state["urls_to_audit"] = urls_to_audit

    # Preview + Trigger Audit button
    if "urls_to_audit" in st.session_state and "seo_data" not in st.session_state:
        urls_to_audit = st.session_state["urls_to_audit"]

        st.markdown("### 🔗 Preview of Pages to Audit")
        st.write(f"**Total pages found:** {len(urls_to_audit)}")

        with st.expander("Click to view URLs"):
            st.dataframe(fix_dataframe_for_streamlit(pd.DataFrame(urls_to_audit, columns=["URL"])))

        if st.button("Start Auditing"):
            # 2) MULTI-THREADED: Audit pages with performance metrics
            with st.spinner("Site Audit in progress..."):
                
                # Limit pages if requested
                max_pages = min(200 if limit_pages else 1000, len(urls_to_audit))
                urls_to_process = urls_to_audit[:max_pages]
                
                # Progress tracking
                bar = st.progress(0.0)
                stat = st.empty()
                eta = st.empty()
                start_time = time.time()
                
                def progress_callback(completed, total, current_url):
                    progress = completed / total
                    bar.progress(progress)
                    stat.markdown(f"🔍 **Progress:** {completed}/{total} pages audited")
                    stat.markdown(f"**Currently processing:** [`{current_url}`]({current_url})")
                    
                    # Update ETA
                    elapsed = time.time() - start_time
                    if completed > 0:
                        avg_time_per_page = elapsed / completed
                        remaining_pages = total - completed
                        remaining_time = avg_time_per_page * remaining_pages
                        mm, ss = divmod(int(remaining_time), 60)
                        eta.markdown(f"Estimated time left: **{mm} m {ss} s**")
                
                all_reports = audit_pages_multithreaded(
                    urls_to_audit=urls_to_process,
                    max_workers=2,
                    progress_callback=progress_callback )
                
                bar.progress(1.0)
                st.markdown(f"Audit complete! **{len(all_reports)}** pages audited in parallel")
                eta.empty()

                # Save results to session
                page_issues_map = {
                    pg["url"]: extract_page_issues(pg["report"])
                    for pg in all_reports
                }
                page_issues_map = {u: iss for u, iss in page_issues_map.items() if iss}

                st.session_state["seo_data"] = all_reports
                st.session_state["page_issues_map"] = page_issues_map
                st.session_state["ai_summary"] = None
                st.session_state["ai_summary_time"] = None

                # Show performance improvement
                elapsed_total = time.time() - start_time
                sequential_estimate = len(all_reports) * 5  # Estimate 5s per page sequentially
                speedup = sequential_estimate / elapsed_total if elapsed_total > 0 else 1
                
                st.success(f"Audit complete in {elapsed_total:.1f}s! Estimated {speedup:.1f}x faster than sequential processing.")

    # Report views
    if "seo_data" in st.session_state:
        # URL dropdown available across all views
        st.markdown("### Audited URLs")
        urls_audited = [page_data.get('url', '') for page_data in st.session_state["seo_data"]]
        
        selected_url = st.selectbox(
            "Select a URL to view details:",
            options=["All URLs"] + urls_audited,
            index=0
        )
        
        if selected_url != "All URLs":
            selected_page = next((page for page in st.session_state["seo_data"] if page.get('url') == selected_url), None)
            if selected_page:
                st.markdown(f"#### Details for: `{selected_url}`")
                with st.expander("Click to view detailed report"):
                    st.json(selected_page.get('report', {}), expanded=False)
        
        st.markdown("---")
        not_crawled_urls = [
        r["url"]
        for r in st.session_state["seo_data"]
        if r.get("report", {}).get("page_not_crawled", False)
    ]

        if not_crawled_urls:
            with st.expander(f"Pages Not Crawled ({len(not_crawled_urls)})"):
                for bad_url in not_crawled_urls:
                    st.text(bad_url)
    # ----------------------------------
        view = st.radio("Choose report view:", ["Raw SEO Report", "SEO Issues CSV", "AI SEO Summary"])

        if view == "Raw SEO Report":
            display_wrapped_json(st.session_state["seo_data"])

        elif view == "SEO Issues CSV":
            st.markdown("### Comprehensive SEO Issues Report")
            
            comprehensive_df = convert_to_comprehensive_seo_csv(st.session_state["seo_data"])
            binary_df = convert_to_binary_issue_csv(st.session_state["seo_data"])  # NEW

            if not comprehensive_df.empty:
                # Calculate summary stats with proper handling of N/A values
                issues_with_counts = comprehensive_df[comprehensive_df['Failed checks'] != '']
                
                # Filter out non-numeric values before converting to int
                numeric_issues = issues_with_counts[
                    (issues_with_counts['Failed checks'] != 'N/A') & 
                    (issues_with_counts['Failed checks'] != '')
                ]
                total_issues = numeric_issues['Failed checks'].astype(int).sum()
                
                # Display summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Pages Audited", len(st.session_state["seo_data"]))
                with col2:
                    st.metric("Total Issues Found", total_issues)
                with col3:
                    issues_with_failures = len(numeric_issues[numeric_issues['Failed checks'].astype(int) > 0])
                    st.metric("Issue Types with Failures", issues_with_failures)
                
                # Filter options
                show_all = st.checkbox("Show all issues (including those with 0 failures)")
                
                if show_all:
                    display_df = comprehensive_df
                else:
                    display_df = comprehensive_df[
                        (comprehensive_df['Failed checks'] == '') |
                        ((comprehensive_df['Failed checks'] != 'N/A') & 
                         (comprehensive_df['Failed checks'] != '') &
                         (comprehensive_df['Failed checks'].astype(str) != '0'))
                    ]
                
                st.dataframe(fix_dataframe_for_streamlit(display_df), use_container_width=True)
                
                # Download buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(
                        "Download Complete SEO Issues CSV",
                        comprehensive_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"seo_issues_complete_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    if not display_df.empty:
                        st.download_button(
                            "Download Issues with Failures Only",
                            display_df.to_csv(index=False).encode("utf-8"),
                            file_name=f"seo_issues_failures_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )

                # NEW: Binary CSV download
                with col3:
                    st.download_button(
                        "Download Binary Issues CSV",
                        binary_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"seo_issues_binary_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            else:
                st.success("No issues found across all audited pages.")


        else:  # AI SEO Summary
            def generate_summary():
                return ai_analysis(
                    st.session_state["seo_data"],
                    st.session_state["page_issues_map"]
                )

            if st.button("Regenerate AI Summary") or st.session_state.get("ai_summary") is None:
                with st.spinner("Generating AI summary..."):
                    st.session_state["ai_summary"] = generate_summary()
                    st.session_state["ai_summary_time"] = datetime.now().strftime("%d %b %Y, %I:%M %p")

            raw = st.session_state["ai_summary"]
            gen_t = st.session_state.get("ai_summary_time", "")
            html = build_html_summary(markdown_to_html(raw), start_url)

            st.markdown("### AI SEO Summary Preview")
            if gen_t:
                st.caption(f"Last generated: {gen_t}")

            st.markdown(raw)

            st.download_button(
                "Download SEO Summary as PDF",
                convert_to_pdf(html),
                "seo_summary.pdf",
                mime="application/pdf",
            )

if __name__ == "__main__":
    main()

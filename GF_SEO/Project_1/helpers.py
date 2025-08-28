import sys
import asyncio
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import random
import time
import tempfile
import shutil
import os
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List
from urllib.parse import urlparse, urljoin
from urllib import robotparser
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import Counter
from threading import Lock
import requests
import pandas as pd
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By


load_dotenv()

# Global cache with thread safety
_http_head_cache: dict[str, int] = {}
_cache_lock = Lock()

# --- Global session with retry and 429-safe backoff ---
_session = requests.Session()
retry_strategy = Retry(
    total=5,  # retry up to 5 times
    backoff_factor=1,  # 1s, 2s, 4s, 8s, ...
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"],
    respect_retry_after_header=True  # KEY for 429s
)
adapter = HTTPAdapter(max_retries=retry_strategy)
_session.mount("http://", adapter)
_session.mount("https://", adapter)

def is_valid_http_url(url: str) -> bool:
    """Check if URL uses HTTP/HTTPS scheme and is valid for requests"""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ('http', 'https') and bool(parsed.netloc)
    except Exception:
        return False

def head_cached(url, timeout=10, retries=2):
    # Skip non-HTTP/HTTPS URLs
    if not is_valid_http_url(url):
        print(f"Skipping non-HTTP/HTTPS URL: {url}")
        return None
        
    for attempt in range(retries + 1):
        try:
            time.sleep(random.uniform(0.5, 1.3))  # throttle
            resp = _session.head(url, timeout=timeout, allow_redirects=True)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 1))
                print(f"429 for {url}, retrying after {retry_after}s")
                time.sleep(retry_after+random.uniform(0.5,1.3))
                continue
            return resp
        except Exception as e:
            print(f"HEAD error for {url}: {e}")
            return None
    return None


import time, random

def head_cached_parallel(urls: list[str], max_workers: int = 3, crawl_delay: float = 0.0) -> dict[str, int]:
    results: dict[str, int] = {}
    to_fetch = []

    # Filter out non-HTTP/HTTPS URLs before processing
    valid_urls = []
    for u in urls:
        if is_valid_http_url(u):
            valid_urls.append(u)
        else:
            print(f"Skipping invalid URL scheme: {u}")
            results[u] = -2  # Special code for invalid URLs

    with _cache_lock:
        for u in valid_urls:
            code = _http_head_cache.get(u)
            if code is not None:
                results[u] = code
            else:
                to_fetch.append(u)

    if to_fetch:
        def _hc(u):
            delay = crawl_delay if crawl_delay > 0 else random.uniform(0.3, 0.8)
            time.sleep(delay)  # jitter to avoid burst
            response = head_cached(u)
            status = response.status_code if response else -1
            return u, status

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for u, code in pool.map(_hc, to_fetch):
                results[u] = code

    return results


# Thread-safe duplicate tracking
class ThreadSafeSets:
    def __init__(self):
        self.titles = set()
        self.descriptions = set()
        self.content_hashes = set()
        self.lock = Lock()
        self.h1_texts = set()
        self.h2_texts = set()
    def check_and_add_title(self, title):
        with self.lock:
            is_duplicate = title in self.titles
            self.titles.add(title)
            return is_duplicate
     
    def check_and_add_description(self, desc):
        with self.lock:
            is_duplicate = desc in self.descriptions
            self.descriptions.add(desc)
            return is_duplicate
    
    def check_and_add_content_hash(self, content_hash):
        with self.lock:
            is_duplicate = content_hash in self.content_hashes
            self.content_hashes.add(content_hash)
            return is_duplicate
    def check_and_add_h1(self, h1_text):
        with self.lock:
            is_dup = h1_text in self.h1_texts
            self.h1_texts.add(h1_text)
            return is_dup
    def check_and_add_h2(self, h2_text):
        with self.lock:
            dup = h2_text in self.h2_texts
            self.h2_texts.add(h2_text)
            return dup

def display_wrapped_json(data: list[dict], preview: int = 10) -> None:
    """Show just the first items of a huge JSON list"""
    if not data:
        st.write("No data to display.")
        return
    
    preview_block = data[:preview]
    st.json(preview_block, expanded=False)
    st.caption(f"Showing the first {len(preview_block)} of {len(data)} pages")
    
    st.download_button(
        "Download full SEO report (JSON)",
        json.dumps(data, indent=2),
        file_name="seo_report.json",
        mime="application/json",
    )

def robust_check_canonical_status(canonical_url, timeout=5):
    """Performs GET with retry on 429 and returns canonical status."""
    # Skip non-HTTP/HTTPS URLs
    if not is_valid_http_url(canonical_url):
        print(f"Skipping non-HTTP canonical URL: {canonical_url}")
        return {"broken": True, "throttled": False, "status": -2}
        
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SEO-AuditBot/1.0)"}
    
    try:
        response = requests.get(canonical_url, headers=headers, timeout=timeout, allow_redirects=True)
        status = response.status_code

        if status == 200:
            return {"broken": False, "throttled": False, "status": 200}
        elif status == 429:
            time.sleep(2)  # Wait and retry
            response = requests.get(canonical_url, headers=headers, timeout=timeout, allow_redirects=True)
            status = response.status_code
            if status == 200:
                return {"broken": False, "throttled": False, "status": 200}
            else:
                return {"broken": False, "throttled": True, "status": status}
        else:
            return {"broken": True, "throttled": False, "status": status}

    except Exception:
        return {"broken": True, "throttled": False, "status": -1}

def should_skip_url(url):
    skip_keywords = [
    "/cart", "/checkout", "/login", "/signin", "/sign-up", "/signup",
    "/register", "/forgot-password", "/reset-password", "/logout",
    "/account", "/accounts", "/my-account", "/user", "/profile",
    "/wishlist", "/favorites", "/compare", "/order", "/orders",
    "/track-order", "/track", "/admin", "/dashboard", "/backend"
]

    if any(kw in url.lower() for kw in skip_keywords):
        return True
    if "?" in url:
        return True
    return False

chrome_semaphore = threading.Semaphore(1)
def robust_page_fetcher(url, wait=0.5):
    """
    Robust page fetcher with fallback strategy and proper resource management
    """
    
    # Method 1: Try simple HTTP request first (faster, more reliable)
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        
        html = response.text
        performance_data = {
            'load_time_seconds': response.elapsed.total_seconds(),
            'page_size_bytes': len(html),
            'method_used': 'requests'
        }
        
        print(f"✅ Successfully fetched {url} via HTTP")
        return html, performance_data
        
    except Exception as e:
        print(f"HTTP request failed for {url}: {e}")
        
        # Method 2: Fallback to ChromeDriver only if necessary
        return _chrome_fallback(url, wait)
    
def convert_to_binary_issue_csv(all_reports):
    """
    Converts detailed page reports into a per-page binary issue matrix CSV.
    Each row = one page
    Each column = one issue
    Values = 1 if issue present, 0 otherwise
    """
    page_rows = []
    all_issues = set()

    for page in all_reports:
        url = page.get("url")
        report = page.get("report", {})
        issues = {}

        # Basic types
        for key, val in report.items():
            if isinstance(val, bool):
                issues[key] = int(val)
            elif isinstance(val, int) and val > 0:
                issues[key] = 1
            elif isinstance(val, list) and len(val) > 0:
                issues[key] = 1
            elif isinstance(val, dict) and key not in {
                "headings", "images", "word_stats", "performance_metrics", "title",
                "description", "robots_txt", "https_info", "schema", "meta_robots", "audit_summary"
            }:
                issues[key] = 1 if len(val) > 0 else 0

        # Special nested checks

        # Headings
        headings = report.get("headings", {})
        if isinstance(headings, dict):
            issues["missing_h1"] = 1 if headings.get("H1", 0) == 0 else 0
            issues["missing_h2"] = 1 if headings.get("H2", 0) == 0 else 0
            issues["multiple_h1"] = 1 if headings.get("H1", 0) > 1 else 0
            issues["h2_sitewide_duplicate"] = 1 if report.get("h2_sitewide_duplicate") else 0
            issues["h2_duplicate"] = 1 if report.get("h2_duplicate") else 0
        # Images
        images = report.get("images", {})
        if isinstance(images, dict):
            issues["missing_image_alt"] = 1 if images.get("images_without_alt", 0) > 0 else 0
            issues["too_many_images"] = 1 if images.get("total_images", 0) > 50 else 0
            issues["broken_internal_images"] = 1 if len(images.get("broken_images", [])) > 0 else 0
            issues["broken_external_images"] = 1 if len(images.get("external_broken_images", [])) > 0 else 0
            issues["image_name_missing"] = 1 if images.get("images_without_filename", 0) > 0 else 0

        # Word stats
        word_stats = report.get("word_stats", {})
        if isinstance(word_stats, dict):
            issues["low_text_to_html_ratio"] = 1 if word_stats.get("anchor_ratio_percent", 100) > 90 or report.get("text_to_html_ratio_percent", 100) < 3 else 0
            issues["too_much_content"] = 1 if word_stats.get("total_words", 0) > 3000 else 0

        # Robots.txt
        robots = report.get("robots_txt", {})
        if isinstance(robots, dict):
            issues["robots_disallows_crawling"] = int(report.get("robots_disallows_crawling", False))
            issues["robots_txt_missing"] = 0 if robots.get("found", False) else 1

        # Schema
        schema = report.get("schema", {})
        if isinstance(schema, dict):
            issues["missing_schema"] = 1 if not (schema.get("json_ld_found") or schema.get("microdata_found")) else 0

        # Performance
        perf = report.get("performance_metrics", {})
        if isinstance(perf, dict):
            issues["slow_page_load"] = 1 if perf.get("load_time_seconds", 0) > 3 else 0
            issues["large_page_size"] = 1 if perf.get("page_size_bytes", 0) > 2_000_000 else 0

        # Meta robots tag
        meta_robots = report.get("meta_robots", "")
        if isinstance(meta_robots, str) and "noindex" in meta_robots.lower():
            issues["page_not_indexed"] = 1

        # Canonical
        if report.get("canonical_pages_broken"):
            issues["canonical_pages_broken"] = 1
        if report.get("canonical_pages_throttled"):
            issues["canonical_pages_throttled"] = 1
        if report.get("canonical_external_domain"):
            issues["canonical_external_domain"] = 1
        if report.get("canonical_status") != 200:
            issues["canonical_status_not_200"] = 1

        # Title
        title = report.get("title", {})
        if isinstance(title, dict):
            issues["missing_meta_title"] = 1 if title.get("text") == "Missing" else 0
            issues["meta_title_over_60_chars"] = 1 if title.get("length", 0) > 60 else 0
            issues["title_too_short"] = 1 if title.get("length", 0) < 10 else 0

        # Description
        description = report.get("description", {})
        if isinstance(description, dict):
            issues["meta_description_too_short"] = 1 if description.get("length", 0) < 50 else 0
            issues["meta_description_over_160_chars"] = 1 if description.get("length", 0) > 160 else 0
            issues["missing_meta_description"] = 1 if description.get("text") == "Missing" else 0
            issues["meta_description_duplicate"] = 1 if description.get("duplicate") else 0

        # URL-specific
        if isinstance(url, str):
            issues["url_over_70_chars"] = 1 if len(url) > 70 else 0
            issues["url_contains_number"] = 1 if any(char.isdigit() for char in url) else 0
            issues["url_contains_symbol"] = 1 if any(c in url for c in ['!', '@', '#', '$', '%', '^', '&', '*']) else 0

        # Additional issue flags
        issues["has_5xx_error"] = 1 if report.get("status_code", 200) >= 500 else 0
        issues["has_4xx_error"] = 1 if 400 <= report.get("status_code", 200) < 500 else 0
        issues["has_3xx_redirect"] = 1 if 300 <= report.get("status_code", 200) < 400 else 0

        # Other expected flags
        issues["nofollow_internal_links"] = 1 if report.get("nofollow_internal_links") else 0
        issues["nofollow_external_links"] = 1 if report.get("nofollow_external_links") else 0
        issues["orphaned_page"] = 1 if report.get("orphaned") else 0
        issues["duplicate_page"] = 1 if report.get("is_duplicate") else 0
        issues["duplicate_content_page"] = 1 if report.get("duplicate_content") else 0
        issues["page_not_crawled"] = 1 if report.get("page_not_crawled") else 0
        issues["crawl_depth_too_long"] = 1 if report.get("crawl_depth", 0) > 4 else 0

        # Finalize
        all_issues.update(issues.keys())
        page_rows.append((url, issues))

    all_issues = sorted(all_issues)
    records = []
    for url, issues in page_rows:
        row = {"Page URL": url}
        for issue in all_issues:
            row[issue] = issues.get(issue, 0)
        records.append(row)

    return pd.DataFrame(records)



def _chrome_fallback(url, wait=0.5):
    """ChromeDriver fallback with proper resource management"""
    
    with chrome_semaphore:  # Ensure only one Chrome instance at a time
        driver = None
        temp_dir = None
        
        try:
            # Create unique temp directory
            temp_dir = tempfile.mkdtemp()
            
            options = uc.ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-images")  # Faster loading
            options.add_argument(f"--user-data-dir={temp_dir}")
            options.add_argument("--single-process")
            options.add_argument("--disable-blink-features=AutomationControlled")
            
            driver = uc.Chrome(options=options, version_main=None)
            driver.set_page_load_timeout(20)
            
            driver.get(url)
            time.sleep(wait)
            
            html = driver.page_source
            performance_data = {
                'load_time_seconds': wait,
                'page_size_bytes': len(html),
                'method_used': 'chrome'
            }
            
            print(f"Successfully rendered {url} via Chrome")
            return html, performance_data
            
        except Exception as e:
            print(f"Chrome fallback failed for {url}: {e}")
            return None, None
            
        finally:
            # Cleanup resources
            if driver:
                try:
                    driver.quit()
                except:
                    pass
            
            if temp_dir:
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass

# Fix DataFrame serialization issues
def fix_dataframe_for_streamlit(df):
    """Convert DataFrame columns to Streamlit-compatible types"""
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].astype(str)
    return df_copy

def get_urls_from_sitemap(sitemap_url):
    """Handles both sitemap.xml and sitemap indexes"""
    EXT_SKIP = {".jpg", ".jpeg", ".png", ".gif", ".webp",
                ".svg", ".pdf", ".zip", ".mp4", ".mov"}

    def fetch_soup(url, retries=3, delay=3):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/115.0.0.0 Safari/537.36"
        }
        for i in range(retries):
            try:
                r = requests.get(url, headers=headers, timeout=10)
                if r.status_code == 429:
                    print(f"⏳ Got 429 Too Many Requests for {url}, retrying in {delay * (i + 1)}s...")
                    time.sleep(delay * (i + 1))
                    continue
                r.raise_for_status()
                return BeautifulSoup(r.content, "xml")
            except Exception as e:
                print(f"Failed to fetch sitemap {url}: {e}")
                return None
        return None

    def is_html_link(link, base_netloc):
        parsed = urlparse(link)
        if parsed.netloc and parsed.netloc != base_netloc:
            return False
        if any(parsed.path.lower().endswith(ext) for ext in EXT_SKIP):
            return False
        return True

    root_soup = fetch_soup(sitemap_url)
    if not root_soup:
        return []

    base_netloc = urlparse(sitemap_url).netloc
    urls = set()

    if root_soup.find("sitemapindex"):
        for sm in root_soup.find_all("sitemap"):
            loc = sm.find("loc")
            if not loc:
                continue
            child_soup = fetch_soup(loc.text.strip())
            if not child_soup:
                continue
            for url_tag in child_soup.find_all("url"):
                loc_tag = url_tag.find("loc")
                if loc_tag and is_html_link(loc_tag.text.strip(), base_netloc):
                    urls.add(loc_tag.text.strip())
    else:
        for url_tag in root_soup.find_all("url"):
            loc_tag = url_tag.find("loc")
            if loc_tag and is_html_link(loc_tag.text.strip(), base_netloc):
                urls.add(loc_tag.text.strip())

    filtered_urls = [
        u for u in urls
        if "/product" not in u.lower() and "/products" not in u.lower() and "/collections" not in u.lower() and "/collection" not in u.lower()  
    ]
    return list(filtered_urls)

def is_supported_http_url(href: str) -> bool:
        parsed = urlparse(href)
        return parsed.scheme in ("http", "https", "")

SOCIAL_DOMAINS = {
            "facebook.com", "instagram.com", "twitter.com", "linkedin.com", "x.com",
            "youtube.com", "pinterest.com", "tiktok.com", "wa.me", "web.whatsapp.com"
        }  # "" = relative link
def is_social_link(href: str) -> bool:
        netloc = urlparse(href).netloc
        return any(domain in netloc for domain in SOCIAL_DOMAINS)

def full_seo_audit(url, shared_sets, html, response_headers=None, response_status=None, performance=None):
    """Complete enhanced SEO audit - Thread-safe version"""
    result = {}
    visited_urls: set[str] = set()
    internal_errors: list[dict] = []

    try:
        if not html:
            result["error"] = f"Could not render page: {url}"
            result["page_not_crawled"] = True
            return result

        soup = BeautifulSoup(html, "html.parser")
        anchor_tags = soup.find_all("a", href=True)
        parsed_url = urlparse(url)

        # Status Code Checks
        if response_status:
            if 500 <= response_status < 600:
                result["status_5xx_error"] = {"status": response_status}
            elif 400 <= response_status < 500:
                result["status_4xx_error"] = {"status": response_status}
            elif 300 <= response_status < 400:
                result["status_3xx_error"] = {"status": response_status}

        # Performance Checks
        if performance:
            load_time = performance.get('load_time_seconds', 0)
            browser_load_time = performance.get('browser_load_time_seconds', load_time)
            actual_load_time = browser_load_time if browser_load_time > 0 else load_time
            
            if actual_load_time > 3.0:
                result["slow_page_load_speed"] = {
                    "load_time": actual_load_time,
                    "threshold": 3.0,
                    "performance_data": performance
                }
            
            if performance.get('page_size_bytes', 0) > 2097152:
                result["large_page_size"] = {
                    "size_bytes": performance['page_size_bytes'],
                    "threshold": 2097152
                }

        # Meta data
        title_tag = soup.find("title")
        desc_tag = soup.find("meta", {"name": "description"})
        title_text = title_tag.text.strip() if title_tag else ""
        desc_text = desc_tag.get("content", "").strip() if desc_tag else ""
        
        if not title_text:
            result["missing_title"] = True
            result["meta_title_missing"] = True
        if not desc_text:
            result["missing_meta_description"] = True
            result["meta_description_missing"] = True

        result["title"] = {
            "text": title_text or "Missing",
            "length": len(title_text),
            "word_count": len(title_text.split()),
        }
        result["description"] = {
            "text": desc_text or "Missing",
            "length": len(desc_text),
            "word_count": len(desc_text.split()),
        }

        # Enhanced title and meta description checks
        if len(title_text) < 30:
            result["title_too_short"] = True
        if len(title_text) > 60:
            result["title_too_long"] = True
            result["meta_title_over_60_chars"] = True
        if len(desc_text) < 120:
            result["meta_description_too_short"] = True
        if len(desc_text) > 160:
            result["meta_description_too_long"] = True

        # Thread-safe duplicate checks
        if shared_sets.check_and_add_title(title_text):
            result["duplicate_title"] = True
            result["meta_title_duplicate"] = True

        if shared_sets.check_and_add_description(desc_text):
            result["duplicate_meta_description"] = True
            result["meta_description_duplicate"] = True

        page_text = " ".join(soup.stripped_strings)
        text_hash = hash(page_text)
        if shared_sets.check_and_add_content_hash(text_hash):
            result["duplicate_content"] = True 
            result["duplicate_content_page"] = True

        # Document structure checks
        if not html.strip().upper().startswith('<!DOCTYPE HTML'):
            result["missing_doctype"] = True

        charset_meta = soup.find("meta", {"charset": True}) or soup.find("meta", {"http-equiv": "Content-Type"})
        if not charset_meta:
            result["missing_charset"] = True

        viewport_meta = soup.find("meta", {"name": "viewport"})
        if not viewport_meta:
            result["missing_viewport"] = True

        html_tag = soup.find("html")
        if not (html_tag and html_tag.get("lang")):
            result["missing_lang_attribute"] = True

        canonical_tags = soup.find_all("link", {"rel": "canonical"})
        if len(canonical_tags) == 0:
            result["missing_canonical"] = True
        elif len(canonical_tags) > 1:
            result["multiple_canonical"] = True
        else:
            canonical_tag = canonical_tags[0]
            canonical_url = canonical_tag.get("href")
            if canonical_url:
                canonical_full_url = urljoin(url, canonical_url)

                # ✅ Use robust checker instead of head_cached_parallel
                check = robust_check_canonical_status(canonical_full_url)
                result["canonical_status"] = check["status"]

                if check["broken"]:
                    result["canonical_pages_broken"] = {
                        "canonical_url": canonical_full_url,
                        "status": check["status"]
                    }

                if check["throttled"]:
                    result["canonical_pages_throttled"] = {
                        "canonical_url": canonical_full_url,
                        "status": check["status"]
                    }

                canonical_domain = urlparse(canonical_full_url).netloc
                if canonical_domain != parsed_url.netloc:
                    result["canonical_external_domain"] = {
                        "canonical_url": canonical_full_url
                    }


        # Enhanced Heading Checks
        headings = {f"H{i}": [h.get_text().strip() for h in soup.find_all(f"h{i}")] for i in range(1, 7)}
        result["headings"] = {f"H{i}": len(headings[f"H{i}"]) for i in range(1, 7)}

        h1_texts = headings["H1"]
        h2_texts = headings["H2"]
        
        # Check if H2s are repeated on the same page
        if len(h2_texts) > len(set(h2_texts)):
            result["h2_duplicate"] = True

        # Check if any H2 is used on other pages
        for txt in set(h2_texts):
            if txt and shared_sets.check_and_add_h2(txt):
                result["h2_sitewide_duplicate"] = True
                break
        h1_text = h1_texts[0] if h1_texts else ""
        result["H1_content"] = h1_text

        if not h1_texts:
            result["missing_h1"] = True
            result["h1_missing"] = True
        if h1_text and shared_sets.check_and_add_h1(h1_text):
            result["h1_sitewide_duplicate"] = True
        if not h2_texts:
            result["missing_h2"] = True
            result["h2_missing"] = True

        if len(h1_texts) > len(set(h1_texts)):
            result["h1_duplicate"] = True

        if len(h2_texts) > len(set(h2_texts)):
            result["h2_duplicate"] = True

        if h1_text and title_text and h1_text.lower() == title_text.lower():
            result["h1_title_duplicate"] = True

        # URL Checks
        if len(url) > 70:
            result["url_too_long"] = {"length": len(url), "threshold": 70}
            result["url_over_70_chars"] = True

        if "_" in parsed_url.path:
            result["url_has_underscores"] = True

        if any(char.isdigit() for char in parsed_url.path):
            result["url_contains_number"] = True

        import string
        allowed_symbols = {'-', '_', '/', '.', '?', '&', '=', '#', ':'}
        url_symbols = set(char for char in url if char in string.punctuation)
        if url_symbols - allowed_symbols:
            result["url_contains_symbol"] = True

        # HTTPS & security checks
        result["https_info"] = {"using_https": url.startswith("https://"), "was_redirected": False}

        if not url.startswith("https://"):
            result["not_using_https"] = True
            result["non_secure_pages"] = True

        # External link checks
        urls_to_test = []
        external_nofollow_links = []

        for a in anchor_tags:
            href = a.get("href")
            if not href or not is_supported_http_url(href):
                continue

            full_url = urljoin(url, href)
            href_domain = urlparse(full_url).netloc

            if href_domain and href_domain != parsed_url.netloc:
                if is_social_link(full_url):
                    continue
                if a.get("rel") and "nofollow" in a.get("rel"):
                    external_nofollow_links.append(full_url)
                # Only add valid HTTP/HTTPS URLs for testing
                if is_valid_http_url(full_url):
                    urls_to_test.append(full_url)

        raw_external = head_cached_parallel(urls_to_test)
        external_broken_links = []
        for u, status in raw_external.items():
            if status >= 400 and status != -2:  # Ignore invalid URL scheme status
                external_broken_links.append({"url": u, "status": status})

        result["external_broken_links"] = external_broken_links
        result["external_nofollow_links"] = external_nofollow_links
    
        # Internal link checks
        internal_nofollow_links = []
        internal_link_count = 0

        for a in anchor_tags:
            href = a.get("href")
            if not href or not is_supported_http_url(href):
                continue
            full_url = urljoin(url, href)
            href_domain = urlparse(full_url).netloc

            if href_domain == parsed_url.netloc or not href_domain:
                internal_link_count += 1
                if a.get("rel") and "nofollow" in a.get("rel"):
                    internal_nofollow_links.append(full_url)

        result["internal_nofollow_links"] = internal_nofollow_links

        total_links = len(anchor_tags)
        if total_links > 100:
            result["too_many_links"] = {"count": total_links, "threshold": 100}

        # Text / anchor stats
        total_words = len(re.findall(r"\b\w+\b", page_text))
        anchor_texts = [a.get_text(strip=True) for a in anchor_tags if a.get_text(strip=True)]
        anchor_words = sum(len(t.split()) for t in anchor_texts)

        result["word_stats"] = {
            "total_words": total_words,
            "anchor_words": anchor_words,
            "anchor_ratio_percent": round((anchor_words / total_words) * 100, 2) if total_words else 0,
            "sample_anchors": anchor_texts[:10],
        }

        result["empty_anchor_text_links"] = sum(1 for a in anchor_tags if not a.get_text(strip=True))

        non_desc = {"click here", "read more", "learn more", "more", "here", "view", "link", "this", "that"}
        result["non_descriptive_anchors"] = sum(1 for t in anchor_texts if t.lower() in non_desc)

        if total_words > 3000:
            result["too_much_content"] = {"word_count": total_words, "threshold": 3000}

        # Enhanced Link Checks
        malformed_links = 0
        
        for a in anchor_tags:
            href = a.get("href", "")
            if href:
                if href.startswith("javascript:") and "void" in href:
                    malformed_links += 1
                elif href.startswith("mailto:") and "@" not in href:
                    malformed_links += 1
                elif href.startswith("#") and len(href) == 1:
                    malformed_links += 1

        if malformed_links > 0:
            result["malformed_links"] = malformed_links

        # Page structure and content checks
        if len(anchor_tags) <= 1:
            result["single_internal_link"] = True

        html_size = len(html)
        result["text_to_html_ratio_percent"] = round((len(page_text) / html_size) * 100, 2) if html_size else 0

        if html_size > 1048576:
            result["html_too_large"] = {"size_bytes": html_size, "threshold": 1048576}

        # Schema markup
        result["schema"] = {
            "json_ld_found": bool(soup.find_all("script", {"type": "application/ld+json"})),
            "microdata_found": bool(soup.find_all(attrs={"itemscope": True})),
        }

        if not result["schema"]["json_ld_found"] and not result["schema"]["microdata_found"]:
            result["no_schema_markup"] = True

        # Enhanced Image Checks
        images = soup.find_all("img")

        if len(images) > 50:
            result["too_many_images"] = {"count": len(images), "threshold": 50}

        images_missing_name = 0
        for img in images:
            src = img.get("src", "")
            if src:
                filename = src.split("/")[-1].split("?")[0]
                if not filename or filename in ["image", "img", "photo", "picture"]:
                    images_missing_name += 1

        if images_missing_name > 0:
            result["images_missing_name"] = images_missing_name

        # Only test valid HTTP/HTTPS image URLs
        image_urls = []
        for img in images[:10]:
            if img.get("src"):
                img_url = urljoin(url, img.get("src"))
                if is_valid_http_url(img_url):
                    image_urls.append(img_url)

        raw = head_cached_parallel(image_urls)
        broken_images = []
        for u, status in raw.items():
            if status >= 400 and status != -2:  # Ignore invalid URL scheme status
                broken_images.append({"src": u, "status": status})

        external_broken_images = []
        for img in images[:20]:
            src = img.get("src")
            if src:
                full_img_url = urljoin(url, src)
                img_domain = urlparse(full_img_url).netloc
                if img_domain != parsed_url.netloc and is_valid_http_url(full_img_url):
                    status_code = head_cached_parallel([full_img_url]).get(full_img_url, 200)
                    if status_code >= 400 and status_code != -2:
                        external_broken_images.append({"src": full_img_url, "status": status_code})

        result["images"] = {
            "total_images": len(images),
            "images_without_alt": sum(1 for img in images if not img.get("alt")),
            "sample_images": [{"src": img.get("src"), "alt": img.get("alt")} for img in images[:5]],
            "broken_images": broken_images,
            "external_broken_images": external_broken_images,
        }

        result["images_without_alt"] = result["images"]["images_without_alt"]
        if broken_images:
            result["broken_images"] = True

        # Robots and meta robots checks
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        result["robots_txt"] = {"found": False, "disallows": []}
        result["disallowed_internal_resource"] = False
        result["robots_disallows_crawling"] = False

        try:
            # Fetch robots.txt content
            robots_response = requests.get(robots_url, timeout=5)
            if robots_response.status_code == 200:
                robots_text = robots_response.text
                # Store raw content and disallow lines (for display)
                disallows = [
                    ln.strip()
                    for ln in robots_text.splitlines()
                    if ln.lower().startswith("disallow")
                ]
                result["robots_txt"] = {
                    "found": True,
                    "disallows": disallows,
                    "content": robots_text,
                }

                # Use robotparser for accurate checking
                rp = robotparser.RobotFileParser()
                rp.parse(robots_text.splitlines())

                # Check if URL is disallowed for generic user-agent
                if not rp.can_fetch("*", url):
                    result["disallowed_internal_resource"] = True

                # Check if crawling is completely disallowed
                # (A plain "/" or "/*" rule at the top)
                if any(rule.strip().lower() in ["disallow: /", "disallow: /*"] for rule in disallows):
                    result["robots_disallows_crawling"] = True
            else:
                result["robots_txt_not_found"] = True
        except Exception:
            result["robots_txt_not_found"] = True

        # Check meta robots tag
        meta_robots = soup.find("meta", {"name": "robots"})
        robots_content = meta_robots.get("content", "") if meta_robots else ""
        result["meta_robots"] = robots_content

        if "noindex" in robots_content.lower():
            result["meta_robots_noindex"] = True
            result["page_not_indexed"] = True

        # Crawl Depth Check
        path_segments = [seg for seg in parsed_url.path.split("/") if seg]
        if len(path_segments) > 3:
            result["page_crawl_depth_too_long"] = {"depth": len(path_segments)}

        # Sitemap checks
        sitemap_url = f"{parsed_url.scheme}://{parsed_url.netloc}/sitemap.xml"
        try:
            sitemap_response = requests.get(sitemap_url, timeout=5)
            if sitemap_response.status_code == 200:
                result["sitemap_found"] = True
            else:
                result["sitemap_found"] = False
                result["sitemap_not_found"] = True
        except Exception:
            result["sitemap_found"] = False
            result["sitemap_not_found"] = True

        # Internal link error probe
        base_domain = parsed_url.netloc
        internal_urls = []
        skip_internal = [
            "/cart", "/checkout", "/account", "/login", "/orders"
        ]
        for a in anchor_tags:
            href = a.get("href")
            if not href:
                continue
            full_u = urljoin(url, href)
            # skip known unimportant paths
            if any(s in full_u.lower() for s in skip_internal):
                continue
            if (urlparse(full_u).netloc                               == base_domain and 
                full_u not in visited_urls and 
                is_valid_http_url(full_u)):
                visited_urls.add(full_u)
                internal_urls.append(full_u)

        for link, code in head_cached_parallel(internal_urls).items():
            # ignore 401, 403, 406 and invalid URL scheme
            if code >= 400 and code not in (401, 403, 406, -2):
                internal_errors.append({"url": link, "status": code})

        result["internal_link_errors"] = internal_errors

        # Final summary stats
        issue_count = sum(1 for key, value in result.items()
                         if key.endswith(('_error', '_errors', '_missing', '_too_short', '_too_long',
                                         '_duplicate', '_broken', '_malformed', '_not_found',
                                         '_disallows', '_noindex', '_used', '_exposed')) and value)

        result["audit_summary"] = {
            "total_checks_performed": len(result),
            "issues_found": issue_count,
            "audit_timestamp": datetime.now().isoformat(),
            "url_audited": url
        }

        if performance:
            result["performance_metrics"] = performance

    except Exception as e:
        result["error"] = str(e)
        result["page_not_crawled"] = True
        return result

    return result

def audit_single_page_worker(url, shared_sets):
    """Worker function for multi-threaded page auditing"""
    try:
        if should_skip_url(url):
            return {"url": url, "report": {"error": "skipped", "page_not_crawled": True}}
        
        html, performance_data = robust_page_fetcher(url)
        if not html:
            return {"url": url, "report": {"error": "render failed", "page_not_crawled": True}}
        
        report = full_seo_audit(url, shared_sets, html, performance=performance_data)
        return {"url": url, "report": report}
        
    except Exception as e:
        return {"url": url, "report": {"error": str(e), "page_not_crawled": True}}

def audit_pages_multithreaded(urls_to_audit, max_workers=2, progress_callback=None):
    """Main multi-threaded auditing function"""
    shared_sets = ThreadSafeSets()
    all_reports = []
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_url = {
            executor.submit(audit_single_page_worker, url, shared_sets): url 
            for url in urls_to_audit
        }
        
        # Process completed jobs
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                all_reports.append(result)
                completed_count += 1
                
                # Progress callback for UI updates
                if progress_callback:
                    progress_callback(completed_count, len(urls_to_audit), url)
                    
            except Exception as e:
                all_reports.append({
                    "url": url, 
                    "report": {"error": f"Worker error: {str(e)}", "page_not_crawled": True}
                })
                completed_count += 1
                
                if progress_callback:
                    progress_callback(completed_count, len(urls_to_audit), url)
    
    return all_reports

def extract_page_issues(report: dict) -> list[str]:
    """Returns a list of plain‑English issue labels"""
    issues = []

    if report.get("title", {}).get("text") == "Missing":
        issues.append("missing title")

    if report.get("description", {}).get("text") == "Missing":
        issues.append("missing meta description")

    if report.get("duplicate_title"):
        issues.append("duplicate title")

    if report.get("duplicate_meta_description"):
        issues.append("duplicate meta description")

    if report.get("duplicate_content"):
        issues.append("duplicate body content")

    if report.get("H1_content", "") == "":
        issues.append("no H1")

    if report.get("h1_sitewide_duplicate"):
        issues.append("H1 duplicated across pages")
    if report.get("h2_sitewide_duplicate"):
        issues.append("H2 duplicated across pages")
    if report.get("h2_duplicate"):
        issues.append("Duplicate H2 tags on this page")    
    if report.get("headings", {}).get("H1", 0) > 1:
        issues.append("multiple H1s")

    if report.get("empty_anchor_text_links", 0):
        issues.append("links with empty anchor text")

    if report.get("external_broken_links"):
        issues.append(f"{len(report['external_broken_links'])} broken outbound link(s)")

    if report.get("images", {}).get("images_without_alt", 0):
        issues.append(f"{report['images']['images_without_alt']} image(s) without alt")

    if report.get("word_stats", {}).get("anchor_ratio_percent", 0) > 15:
        issues.append("anchor‑text ratio > 15 %")

    if report.get("text_to_html_ratio_percent", 100) < 3:
        issues.append("text‑to‑HTML ratio < 3 %")

    if not report.get("schema", {}).get("json_ld_found"):
        issues.append("no JSON‑LD schema")

    return issues

def convert_to_comprehensive_seo_csv(seo_data: list[dict]) -> pd.DataFrame:
    """Convert to comprehensive SEO issues with ALL 50 specified issues"""
    issue_counts = Counter()
    total_pages = len(seo_data)
    
    # Proper tracking variables
    pages_with_images = 0
    pages_with_titles = 0
    pages_with_descriptions = 0
    pages_with_h1 = 0
    pages_with_h2 = 0
    unique_titles = set()
    unique_descriptions = set()
    unique_h1s = set()
    unique_h2s = set()
    
    # Site-level checks
    site_has_sitemap = True
    site_has_robots = True
    
    for page_data in seo_data:
        report = page_data.get('report', {})
        url = page_data.get('url', '')
        
        # Track content types for proper total calculations
        title_text = report.get('title', {}).get('text', '').strip()
        desc_text = report.get('description', {}).get('text', '').strip()
        h1_text = report.get('H1_content', '').strip()
        
        if title_text and title_text != 'Missing':
            pages_with_titles += 1
            unique_titles.add(title_text)
        if desc_text and desc_text != 'Missing':
            pages_with_descriptions += 1
            unique_descriptions.add(desc_text)
        if h1_text:
            pages_with_h1 += 1
            unique_h1s.add(h1_text)
        
        headings = report.get('headings', {})
        if headings.get('H2', 0) > 0:
            pages_with_h2 += 1
            unique_h2s.add(f"{url}_h2")
        
        images_data = report.get('images', {})
        if images_data.get('total_images', 0) > 0:
            pages_with_images += 1
        
        # Site-level checks
        if not report.get('sitemap_found', True) or report.get('sitemap_not_found'):
            site_has_sitemap = False
        if report.get('robots_txt_not_found'):
            site_has_robots = False
        
        # Count all issues
        if report.get('status_5xx_error'): 
            issue_counts['5xx_errors'] += 1
        if report.get('status_4xx_error'): 
            issue_counts['4xx_errors'] += 1
        if report.get('status_3xx_error') or report.get('redirect_analysis'): 
            issue_counts['3xx_errors'] += 1
            
        # Meta Title issues
        if report.get('missing_title') or report.get('meta_title_missing'): 
            issue_counts['meta_title_missing'] += 1
        if report.get('title_too_long') or report.get('meta_title_over_60_chars') or (len(title_text) > 60): 
            issue_counts['meta_title_over_60_chars'] += 1
        if report.get('duplicate_title') or report.get('meta_title_duplicate'): 
            issue_counts['meta_title_duplicate'] += 1
            
        # Meta Description issues
        if report.get('missing_meta_description') or report.get('meta_description_missing'): 
            issue_counts['meta_description_missing'] += 1
        if report.get('meta_description_too_long') or report.get('meta_description_over_160_chars') or (len(desc_text) > 160): 
            issue_counts['meta_description_over_160_chars'] += 1
        if report.get('duplicate_meta_description') or report.get('meta_description_duplicate'): 
            issue_counts['meta_description_duplicate'] += 1
            
        # Heading issues
        if report.get('missing_h1') or report.get('h1_missing'): 
            issue_counts['h1_missing'] += 1
        if report.get('h1_sitewide_duplicate'): 
            issue_counts['h1_sitewide_duplicate'] += 1
        if report.get('multiple_h1'): 
            issue_counts['multiple_h1'] += 1
        if report.get('missing_h2') or report.get('h2_missing'): 
            issue_counts['h2_missing'] += 1    
        if report.get('h2_duplicate'): 
            issue_counts['h2_duplicate'] += 1
        if report.get('h2_sitewide_duplicate'): 
            issue_counts['h2_sitewide_duplicate'] += 1
        # Image issues
        if report.get('images_missing_name', 0) > 0: 
            issue_counts['image_name_missing'] += 1
        if report.get('images_without_alt', 0) > 0 or images_data.get('images_without_alt', 0) > 0: 
            issue_counts['image_alt_missing'] += 1
        if report.get('broken_images') or len(images_data.get('broken_images', [])) > 0: 
            issue_counts['internal_image_broken'] += 1
        if len(images_data.get('external_broken_images', [])) > 0: 
            issue_counts['external_image_broken'] += 1
                
        # Link issues
        if len(report.get('internal_link_errors', [])) > 0: 
            issue_counts['internal_link_broken'] += 1
        if len(report.get('external_broken_links', [])) > 0: 
            issue_counts['external_link_broken'] += 1
        if len(report.get('internal_nofollow_links', [])) > 0: 
            issue_counts['nofollow_internal'] += 1
        if len(report.get('external_nofollow_links', [])) > 0: 
            issue_counts['nofollow_external'] += 1
        if report.get('empty_anchor_text_links', 0) > 0: 
            issue_counts['no_anchor_text'] += 1
        if report.get('non_descriptive_anchors', 0) > 0: 
            issue_counts['non_descriptive_anchor'] += 1
        if report.get('malformed_links', 0) > 0: 
            issue_counts['malformed_link'] += 1
            
        # URL issues
        if report.get('url_over_70_chars') or len(url) > 70: 
            issue_counts['url_over_70_chars'] += 1
        if report.get('url_contains_number') or any(char.isdigit() for char in urlparse(url).path): 
            issue_counts['url_contains_number'] += 1
        if report.get('url_contains_symbol') or any(symbol in url for symbol in ['%', '&', '=', '?', '#']): 
            issue_counts['url_contains_symbol'] += 1
            
        # Other technical issues
        if (report.get('missing_canonical') or report.get('canonical_pages_broken') or 
            report.get('multiple_canonical')): 
            issue_counts['canonical_pages'] += 1
        if report.get('duplicate_content') or report.get('duplicate_content_page'): 
            issue_counts['duplicate_page'] += 1
        if report.get('no_schema_markup'): 
            issue_counts['schema_missing'] += 1
        if report.get('duplicate_content_page'): 
            issue_counts['duplicate_content_page'] += 1
        if report.get('disallowed_internal_resource'): 
            issue_counts['disallowed_internal_resource'] += 1
        if report.get('slow_page_load_speed'): 
            issue_counts['slow_page_load_speed'] += 1
        if report.get('page_not_crawled'): 
            issue_counts['page_not_crawled'] += 1
        if report.get('page_not_indexed') or report.get('meta_robots_noindex'): 
            issue_counts['page_not_indexed'] += 1
        if report.get('page_crawl_depth_too_long'): 
            issue_counts['page_crawl_depth_too_long'] += 1
        if report.get('text_to_html_ratio_percent', 100) < 3: 
            issue_counts['low_text_html_ratio'] += 1

    def safe_percentage(count, total):
        if total <= 0:
            return "0.0%"
        percentage = min((count / total) * 100, 100.0)
        return f"{percentage:.1f}%"

    # Set proper totals for different categories
    duplicate_title_total = max(pages_with_titles, 1)
    duplicate_desc_total = max(pages_with_descriptions, 1)
    duplicate_h1_total = max(pages_with_h1, 1)
    duplicate_h2_total = max(pages_with_h2, 1)
    images_total = max(pages_with_images, 1)
    
    # Add site-level issue counts
    if not site_has_sitemap:
        issue_counts['sitemap_not_found'] = 1
    if not site_has_robots:
        issue_counts['robots_txt_not_found'] = 1

    # Create complete CSV with all 50 issues
    issues_data = [
        # ERRORS CATEGORY
        {'Issue': 'Errors', 'Failed checks': '', 'Total checks': '', 'Percentage': ''},
        {'Issue': '5xx errors', 'Failed checks': issue_counts.get('5xx_errors', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('5xx_errors', 0), total_pages)},
        {'Issue': '4xx errors', 'Failed checks': issue_counts.get('4xx_errors', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('4xx_errors', 0), total_pages)},
        {'Issue': '3xx errors', 'Failed checks': issue_counts.get('3xx_errors', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('3xx_errors', 0), total_pages)},
        
        # META TITLE CATEGORY
        {'Issue': 'Meta Title', 'Failed checks': '', 'Total checks': '', 'Percentage': ''},
        {'Issue': 'Meta Title is missing', 'Failed checks': issue_counts.get('meta_title_missing', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('meta_title_missing', 0), total_pages)},
        {'Issue': 'Meta Title is over 60 characters', 'Failed checks': issue_counts.get('meta_title_over_60_chars', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('meta_title_over_60_chars', 0), total_pages)},
        {'Issue': 'Meta Title is duplicate', 'Failed checks': issue_counts.get('meta_title_duplicate', 0), 'Total checks': duplicate_title_total, 'Percentage': safe_percentage(issue_counts.get('meta_title_duplicate', 0), duplicate_title_total)},
        
        # META DESCRIPTION CATEGORY
        {'Issue': 'Meta Description', 'Failed checks': '', 'Total checks': '', 'Percentage': ''},
        {'Issue': 'Meta Description is missing', 'Failed checks': issue_counts.get('meta_description_missing', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('meta_description_missing', 0), total_pages)},
        {'Issue': 'Meta Description over 160 characters', 'Failed checks': issue_counts.get('meta_description_over_160_chars', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('meta_description_over_160_chars', 0), total_pages)},
        {'Issue': 'Meta Description is duplicate', 'Failed checks': issue_counts.get('meta_description_duplicate', 0), 'Total checks': duplicate_desc_total, 'Percentage': safe_percentage(issue_counts.get('meta_description_duplicate', 0), duplicate_desc_total)},
        
        # HEADINGS CATEGORY
        {'Issue': 'Headings', 'Failed checks': '', 'Total checks': '', 'Percentage': ''},
        {'Issue': 'H1 is missing', 'Failed checks': issue_counts.get('h1_missing', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('h1_missing', 0), total_pages)},
        {'Issue': 'H2 is missing', 'Failed checks': issue_counts.get('h2_missing', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('h2_missing', 0), total_pages)},
        {'Issue': 'H1 is duplicate across pages', 'Failed checks': issue_counts.get('h1_sitewide_duplicate', 0), 'Total checks': duplicate_h1_total, 'Percentage': safe_percentage(issue_counts.get('h1_sitewide_duplicate', 0), duplicate_h1_total)},
        {'Issue': 'H2 is duplicate', 'Failed checks': issue_counts.get('h2_duplicate', 0), 'Total checks': duplicate_h2_total, 'Percentage': safe_percentage(issue_counts.get('h2_duplicate', 0), duplicate_h2_total)},
        {'Issue': 'H2 is duplicate across pages', 'Failed checks': issue_counts.get('h2_sitewide_duplicate', 0), 'Total checks': duplicate_h2_total, 'Percentage': safe_percentage(issue_counts.get('h2_sitewide_duplicate', 0), duplicate_h2_total)},

        # IMAGES CATEGORY
        {'Issue': 'Images', 'Failed checks': '', 'Total checks': '', 'Percentage': ''},
        {'Issue': 'Image name is missing', 'Failed checks': issue_counts.get('image_name_missing', 0), 'Total checks': images_total, 'Percentage': safe_percentage(issue_counts.get('image_name_missing', 0), images_total)},
        {'Issue': 'Image Alt text is missing', 'Failed checks': issue_counts.get('image_alt_missing', 0), 'Total checks': images_total, 'Percentage': safe_percentage(issue_counts.get('image_alt_missing', 0), images_total)},
        {'Issue': 'Internal image is broken', 'Failed checks': issue_counts.get('internal_image_broken', 0), 'Total checks': images_total, 'Percentage': safe_percentage(issue_counts.get('internal_image_broken', 0), images_total)},
        {'Issue': 'External image is broken', 'Failed checks': issue_counts.get('external_image_broken', 0), 'Total checks': images_total, 'Percentage': safe_percentage(issue_counts.get('external_image_broken', 0), images_total)},
        
        # LINK CATEGORY
        {'Issue': 'Link', 'Failed checks': '', 'Total checks': '', 'Percentage': ''},
        {'Issue': 'Internal link is broken', 'Failed checks': issue_counts.get('internal_link_broken', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('internal_link_broken', 0), total_pages)},
        {'Issue': 'External link is broken', 'Failed checks': issue_counts.get('external_link_broken', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('external_link_broken', 0), total_pages)},
        {'Issue': 'Nofollow attributes in internal link', 'Failed checks': issue_counts.get('nofollow_internal', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('nofollow_internal', 0), total_pages)},
        {'Issue': 'Nofollow attributes in external link', 'Failed checks': issue_counts.get('nofollow_external', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('nofollow_external', 0), total_pages)},
        {'Issue': 'Link with no anchor text', 'Failed checks': issue_counts.get('no_anchor_text', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('no_anchor_text', 0), total_pages)},
        {'Issue': 'Link with non-descriptive anchor text', 'Failed checks': issue_counts.get('non_descriptive_anchor', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('non_descriptive_anchor', 0), total_pages)},
        {'Issue': 'Link is malformed', 'Failed checks': issue_counts.get('malformed_link', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('malformed_link', 0), total_pages)},
        
        # URL CATEGORY
        {'Issue': 'URL', 'Failed checks': '', 'Total checks': '', 'Percentage': ''},
        {'Issue': 'URL is over 70 characters', 'Failed checks': issue_counts.get('url_over_70_chars', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('url_over_70_chars', 0), total_pages)},
        {'Issue': 'URL contains a number', 'Failed checks': issue_counts.get('url_contains_number', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('url_contains_number', 0), total_pages)},
        {'Issue': 'URL contains a symbol', 'Failed checks': issue_counts.get('url_contains_symbol', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('url_contains_symbol', 0), total_pages)},
        
        # OTHER ISSUES CATEGORY
        {'Issue': 'Other Issues', 'Failed checks': '', 'Total checks': '', 'Percentage': ''},
        {'Issue': 'Orphaned pages', 'Failed checks': 0, 'Total checks': total_pages, 'Percentage': '0.0%'},
        {'Issue': 'Canonical pages', 'Failed checks': issue_counts.get('canonical_pages', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('canonical_pages', 0), total_pages)},
        {'Issue': 'Duplicate page', 'Failed checks': issue_counts.get('duplicate_page', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('duplicate_page', 0), total_pages)},
        {'Issue': 'Schema missing', 'Failed checks': issue_counts.get('schema_missing', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('schema_missing', 0), total_pages)},
        {'Issue': 'Duplicate content page', 'Failed checks': issue_counts.get('duplicate_content_page', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('duplicate_content_page', 0), total_pages)},
        {'Issue': 'Disallowed internal resource', 'Failed checks': issue_counts.get('disallowed_internal_resource', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('disallowed_internal_resource', 0), total_pages)},
        {'Issue': 'Slow page load speed', 'Failed checks': issue_counts.get('slow_page_load_speed', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('slow_page_load_speed', 0), total_pages)},
        {'Issue': 'Page not crawled', 'Failed checks': issue_counts.get('page_not_crawled', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('page_not_crawled', 0), total_pages)},
        {'Issue': 'Page not indexed', 'Failed checks': issue_counts.get('page_not_indexed', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('page_not_indexed', 0), total_pages)},
        {'Issue': 'Page crawl depth is too long', 'Failed checks': issue_counts.get('page_crawl_depth_too_long', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('page_crawl_depth_too_long', 0), total_pages)},
        {'Issue': 'Low text to HTML ratio', 'Failed checks': issue_counts.get('low_text_html_ratio', 0), 'Total checks': total_pages, 'Percentage': safe_percentage(issue_counts.get('low_text_html_ratio', 0), total_pages)},
        {'Issue': 'Sitemap.xml not found', 'Failed checks': issue_counts.get('sitemap_not_found', 0), 'Total checks': 1, 'Percentage': safe_percentage(issue_counts.get('sitemap_not_found', 0), 1)},
        {'Issue': 'Robots.txt not found', 'Failed checks': issue_counts.get('robots_txt_not_found', 0), 'Total checks': 1, 'Percentage': safe_percentage(issue_counts.get('robots_txt_not_found', 0), 1)},
    ]
    
    return pd.DataFrame(issues_data)

def ai_analysis(report, page_issues_map=None):
    issues_block = "\n".join(
        f"- **{url}**: " + ", ".join(problems)
        for url, problems in page_issues_map.items()
    ) or "✅ No page‑level issues detected."

    prompt = f"""You are an advanced SEO and web performance analyst. I am providing a JSON-formatted audit report of a website. This JSON includes data for individual URLs covering:

- HTTP/HTTPS status and response codes (including 4xx and 5xx errors)
- Page speed and response time
- Metadata (title, description, length, duplication)
- Content elements (word count, heading structure, text-to-HTML ratio)
- Link data (internal/external links, anchor text quality, redirects)
- Image data (alt tag presence, broken images)
- Schema markup presence
- Indexing and crawling restrictions (robots.txt, meta robots)

Your response should follow this structure:

### 🧠 AI-Powered SEO Summary

Then provide a detailed analysis, structured into these sections:

1. **Overall Health Summary**
Brief summary of the site's technical SEO status.

2. **Strengths**
Highlight technical strengths (e.g. HTTPS, schema usage, fast load times).

3. **Issues to Fix**
Include only issues that are detected in the audit report.

4. **Critical Page-Level Errors**
List problematic URLs and their specific technical issues.

5. **Actionable Recommendations**
Give clear steps to improve technical SEO, indexing, crawlability, and UX.

---

Important:
- Parse the full report without skipping fields.
- Do NOT return your output as JSON.
- Do NOT include triple backticks or code blocks.
- Make the response client-friendly, as if it's going into a formal audit report.
- Maintain clean structure, use bullet points and sections for clarity.

issues_block: {issues_block}

[SEO_REPORT]: {report}
"""

    api_key = os.getenv("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Error during Gemini API call: {e}\n\nDetails: {response.text}"

# --- IMPORTS ---
import os
import time
import json
import requests
import warnings
import ssl
import re
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter
from urllib.parse import urlparse, urljoin, quote_plus

import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai

import nltk
import textstat
import extruct
from w3lib.html import get_base_url
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- NLTK DATA DOWNLOAD ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

resources_to_download = [
    ("punkt", "tokenizers/punkt"),
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
    ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger")
]
for resource_id, path in resources_to_download:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource_id, quiet=True)

# --- CONTINUED IMPORTS (DEPENDENT ON NLTK DATA) ---
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI SEO Analyst",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GLOBAL CONFIGURATION ---
warnings.filterwarnings('ignore')

CONFIG = {
    "app": {
        "title": "AI SEO Analyst",
        "icon": "🔮",
        "version": "22.0", # Updated Version
    },
    "theme": {
        "bg_color": "#f0f2f6",
        "text_color": "#000000",
        "card_bg": "#ffffff",
        "accent_color": "#0ea5e9",
        "primary_color": "#3b82f6"
    },
    "seo_scores": {
        "weights": {
            'title_ok': 10, 'desc_ok': 10, 'h1_ok': 10, 'https': 10,
            'content_long': 10, 'alt_tags_ok': 10, 'readability_ok': 5,
            'canonical': 5, 'schema': 5, 'viewport': 5, 'internal_links': 5,
            'load_time_ok': 5
        },
        "title_range": (50, 60),
        "desc_range": (120, 158),
        "min_word_count": 300,
        "min_readability": 60,
        "max_load_time": 3.0,
        "min_internal_links": 3
    },
    "api": {
        "user_agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        "timeout": 15
    }
}

# --- INITIALIZATION & SETUP ---
@st.cache_resource
def configure_ai():
    """Configures the Gemini API key and tests the connection."""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            model.generate_content("test", request_options={'timeout': 10})
            return True
        except Exception as e:
            st.error(f"Failed to configure AI or connect to Gemini API. Please check your `GEMINI_API_KEY`. Error: {e}", icon="🔑")
            return False
    st.warning("Gemini API Key not found. Please add it to your `.streamlit/secrets.toml` file.", icon="🔑")
    return False

def init_session_state():
    """Initializes session state variables."""
    defaults = {
        'analysis_results': None, 'page': "🏠 Home", 'copilot_history': [],
        'keyword_results': None, 'rewriter_results': None, 'backlink_results': None,
        'semantic_results': None, 'structure_results': None, 'competitor_results': None,
        'api_key_configured': False, 'backlink_target_domain': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- CORE LOGIC: SEO AUDITOR CLASS ---
class SEOAuditor:
    def __init__(self, url: Optional[str] = None, html_content: Optional[str] = None):
        self.url = url; self.html_content = html_content; self.results: Dict[str, Any] = {}; self.soup: Optional[BeautifulSoup] = None; self.headers = {'User-Agent': CONFIG['api']['user_agent']}; self.perf_data = {}; self.lemmatizer = WordNetLemmatizer(); self.stop_words_set = set(stopwords.words('english'))
    def run_audit(self) -> Optional[Dict[str, Any]]:
        try:
            if not self._fetch_and_parse_html(): return None
            self.results = {'performance': self.perf_data, 'on_page': self._check_on_page(), 'content': self._analyze_content(), 'technical': self._check_technical(), 'url': self.url}; self.results['seo_score'] = self._calculate_seo_score(); return self.results
        except Exception as e: st.error(f"An unexpected error occurred during analysis: {e}", icon="🚨"); return None
    def _fetch_and_parse_html(self) -> bool:
        try:
            if self.url:
                if not (self.url.startswith('http://') or self.url.startswith('https://')): raise ValueError("Invalid URL. Please include http:// or https://")
                start_time = time.time(); response = requests.get(self.url, headers=self.headers, timeout=CONFIG['api']['timeout']); response.raise_for_status(); self.html_content = response.text; self.perf_data = {'load_time': time.time() - start_time, 'size_kb': len(response.content) / 1024}
            elif self.html_content: self.url = "Pasted Content"; self.perf_data = {'load_time': 'N/A', 'size_kb': len(self.html_content.encode('utf-8')) / 1024}
            else: raise ValueError("Either a URL or HTML content must be provided.")
            self.soup = BeautifulSoup(self.html_content, 'html.parser')
            for element in self.soup(["script", "style", "noscript", "iframe", "header", "footer", "nav", "aside", "form"]): element.decompose()
            return True
        except requests.exceptions.RequestException as e: st.error(f"Network Error: Could not retrieve URL. {e}", icon="🌐"); return False
        except ValueError as e: st.warning(str(e), icon="⚠️"); return False
    def _check_on_page(self) -> Dict[str, Any]:
        title_tag = self.soup.find('title'); title_text = title_tag.string.strip() if title_tag and title_tag.string else "Not Found"
        desc_tag = self.soup.find('meta', attrs={'name': 'description'}); desc_text = desc_tag.get('content', 'Not Found').strip() if desc_tag else "Not Found"
        h1_tags = self.soup.find_all('h1')
        return {'title': {'text': title_text, 'length': len(title_text), 'status': CONFIG['seo_scores']['title_range'][0] <= len(title_text) <= CONFIG['seo_scores']['title_range'][1]},'description': {'text': desc_text, 'length': len(desc_text), 'status': CONFIG['seo_scores']['desc_range'][0] <= len(desc_text) <= CONFIG['seo_scores']['desc_range'][1]},'headings': {f'h{i}': [h.get_text(strip=True) for h in self.soup.find_all(f'h{i}')] for i in range(1, 7)},'multiple_h1': len(h1_tags) > 1, 'h1_present': len(h1_tags) > 0, 'image_alt_status': self._check_image_alt_text()}
    def _analyze_content(self) -> Dict[str, Any]:
        main_content_area = self.soup.find('main') or self.soup.find('article') or self.soup.body; text = main_content_area.get_text(separator=' ', strip=True) if main_content_area else ""
        words = [self.lemmatizer.lemmatize(word) for word in word_tokenize(text.lower()) if word.isalpha()]; filtered_words = [word for word in words if word not in self.stop_words_set]
        images = self.soup.find_all('img'); total_img, alt_img = len(images), sum(1 for img in images if img.get('alt', '').strip())
        return {'raw_text': text, 'word_count': len(words), 'readability_score': textstat.flesch_reading_ease(text) if text else 0,'keywords': Counter(filtered_words).most_common(20), 'images': {'total': total_img, 'with_alt': alt_img, 'missing_alt': total_img - alt_img}}
    def _check_image_alt_text(self) -> str:
        images = self.soup.find_all('img')
        if not images: return "No images found on the page."
        missing_alt_count = sum(1 for img in images if not img.get('alt', '').strip()); total_images = len(images)
        if missing_alt_count == 0: return "Excellent! All images have alt text."
        else: return f"{missing_alt_count} of {total_images} images are missing alt text."
    def _check_technical(self) -> Dict[str, Any]:
        base_url = get_base_url(self.html_content, self.url) if self.url != "Pasted Content" else ""; domain = urlparse(base_url).netloc; internal_links_count = 0
        if domain:
            for link in self.soup.find_all('a', href=True):
                full_link = urljoin(base_url, link['href'])
                if full_link.startswith(('http://', 'https://')) and domain in urlparse(full_link).netloc: internal_links_count += 1
        robots_txt_present = False; sitemap_xml_present = False
        if self.url and self.url != "Pasted Content":
            root_domain = f"{urlparse(self.url).scheme}://{urlparse(self.url).netloc}"
            try:
                robots_response = requests.get(urljoin(root_domain, '/robots.txt'), headers=self.headers, timeout=5)
                if robots_response.status_code == 200 and "user-agent" in robots_response.text.lower(): robots_txt_present = True
            except requests.exceptions.RequestException: pass
            try:
                sitemap_response = requests.get(urljoin(root_domain, '/sitemap.xml'), headers=self.headers, timeout=5)
                if sitemap_response.status_code == 200 and "<urlset" in sitemap_response.text.lower(): sitemap_xml_present = True
            except requests.exceptions.RequestException: pass
        viewport_tag = self.soup.find('meta', attrs={'name': 'viewport'}); mobile_friendly = bool(viewport_tag and 'width=device-width' in viewport_tag.get('content', ''))
        return {'https': self.url.startswith('https://') if self.url != "Pasted Content" else False,'canonical': bool(self.soup.find('link', rel='canonical')),'schema_present': bool(extruct.extract(self.html_content, syntaxes=['json-ld'])), 'viewport': bool(viewport_tag),'internal_links_count': internal_links_count, 'robots_txt_present': robots_txt_present,'sitemap_xml_present': sitemap_xml_present, 'mobile_friendly_meta': mobile_friendly}
    def _calculate_seo_score(self) -> int:
        score, weights = 0, CONFIG['seo_scores']['weights']; total_weight, res, s_conf = sum(weights.values()), self.results, CONFIG['seo_scores']
        checks = {'title_ok': res['on_page']['title']['status'], 'desc_ok': res['on_page']['description']['status'],'h1_ok': len(res['on_page']['headings']['h1']) == 1, 'https': res['technical']['https'],'content_long': res['content']['word_count'] >= s_conf['min_word_count'], 'alt_tags_ok': res['content']['images']['missing_alt'] == 0,'readability_ok': res['content']['readability_score'] >= s_conf['min_readability'], 'canonical': res['technical']['canonical'],'schema': res['technical']['schema_present'], 'viewport': res['technical']['viewport'],'internal_links': res['technical'].get('internal_links_count', 0) >= s_conf['min_internal_links'],'load_time_ok': isinstance(res['performance']['load_time'], float) and res['performance']['load_time'] < s_conf['max_load_time']}
        for check, is_ok in checks.items():
            if is_ok: score += weights.get(check, 0)
        return int((score / total_weight) * 100) if total_weight > 0 else 0

# --- FEATURE IMPLEMENTATIONS ---
def extract_text_from_url(url: str) -> Optional[str]:
    try:
        headers = {'User-Agent': CONFIG['api']['user_agent']}; response = requests.get(url, headers=headers, timeout=CONFIG['api']['timeout']); response.raise_for_status(); soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav", "aside", "form"]): element.decompose()
        main_content_area = soup.find('main') or soup.find('article') or soup.body
        return main_content_area.get_text(separator=' ', strip=True) if main_content_area else ""
    except: return None
def perform_semantic_analysis(text: str, model, target_keywords: Optional[List[str]] = None) -> Dict[str, Any]:
    if not text: return {"summary": "No content to analyze.", "entities": [], "lsi_keywords": [], "readability_score": 0}
    readability_score = textstat.flesch_reading_ease(text); prompt_parts = ["As an expert SEO analyst, analyze the following text for semantic SEO. Provide:", "1. A brief summary of the text's core topic.", "2. A list of key entities (people, places, organizations, concepts) mentioned.", "3. A list of 5-10 suggested LSI (Latent Semantic Indexing) keywords that are contextually related."]; keys_for_json = "'summary', 'entities', 'lsi_keywords'"
    if target_keywords: prompt_parts.append(f"4. Additionally, check if the following target keywords are covered well in the text: {', '.join(target_keywords)}. Comment on their presence and semantic relevance."); keys_for_json += ", 'keyword_coverage'"
    prompt_parts.extend([f"\nPresent the output as a single, valid JSON object with the following keys ONLY: {keys_for_json}.", f"\nText to analyze:\n{text[:4000]}"]); prompt = "\n".join(prompt_parts)
    try:
        response = model.generate_content(prompt); json_response = response.text.strip().lstrip("```json").rstrip("```"); parsed_results = json.loads(json_response); parsed_results['readability_score'] = readability_score
        return parsed_results
    except: return {"summary": "AI analysis failed.", "entities": [], "lsi_keywords": [], "readability_score": readability_score}
def generate_structure_suggestions(text: str, model) -> str:
    if not text: return "No content provided for structure optimization."
    prompt = f"As a content editor specializing in on-page SEO, analyze the structure of the following text.\nProvide a rewritten version of the text with an optimized structure...\n\nText to optimize:\n{text[:5000]}"
    try: return model.generate_content(prompt).text
    except Exception as e: st.error(f"Failed to generate structure suggestions with AI: {e}"); return "AI structure optimization failed."
def get_serp_competitor_urls(keyword: str, num_results: int = 5) -> List[str]:
    search_url = f"https://www.google.com/search?q={quote_plus(keyword)}"; headers = {'User-Agent': CONFIG['api']['user_agent']}; competitor_urls = []
    try:
        response = requests.get(search_url, headers=headers, timeout=CONFIG['api']['timeout']); response.raise_for_status(); soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('/url?q='):
                actual_url = href.split('/url?q=')[1].split('&')[0]
                if urlparse(actual_url).netloc and "google.com" not in actual_url and actual_url not in competitor_urls: competitor_urls.append(actual_url)
                if len(competitor_urls) >= num_results: break
        if not competitor_urls: st.warning(f"Could not find organic search results for '{keyword}'.")
        return competitor_urls
    except: st.error(f"Failed to scrape SERP for '{keyword}'."); return []
def get_content_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2: return 0.0
    lemmatizer = WordNetLemmatizer()
    def custom_tokenizer(text): return [lemmatizer.lemmatize(w) for w in word_tokenize(text.lower()) if w.isalpha()]
    vectorizer = TfidfVectorizer(stop_words='english', tokenizer=custom_tokenizer)
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2]); similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(f"{similarity:.2f}")
    except: return 0.0

# --- UI & PAGE RENDERING ---
def load_css():
    theme = CONFIG['theme']; st.markdown(f"""<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');html, body, [class*="st-"] {{ font-family: 'Inter', sans-serif; color: {theme['text_color']}; }}.stApp {{ background-color: {theme['bg_color']}; }}div[data-testid="stSidebar"] > div:first-child {{ background-color: {theme['card_bg']}; }}.stButton>button {{ border-radius: 0.5rem; }}div[data-testid="stFormSubmitButton"] > button {{background: linear-gradient(135deg, {theme['primary_color']} 0%, {theme['accent_color']} 100%);color: white; border: none;}}</style>""", unsafe_allow_html=True)
def display_check(label: str, is_ok: bool, help_text: str = ""):
    status_span = f'<span style="color: {"#16a34a" if is_ok else "#dc2626"};">{"✅ OK" if is_ok else "❌ Needs Improvement"}</span>'
    st.markdown(f"**{label}:** {status_span}", unsafe_allow_html=True, help=help_text)
def home_page():
    st.title(f"{CONFIG['app']['icon']} {CONFIG['app']['title']}"); st.markdown("Enter a URL for an instant, in-depth SEO audit.")
    with st.form(key="analysis_form"):
        url_input = st.text_input("Your Website URL", placeholder="https://www.yourdomain.com/page-to-analyze")
        with st.expander("Advanced Options"): html_input = st.text_area("Or Paste Raw HTML", height=150); target_keywords_input = st.text_input("Target Keywords (comma-separated)"); competitor_urls_input = st.text_area("Competitor URLs (one per line)")
        if st.form_submit_button("🚀 Perform All Analyses", type="primary", use_container_width=True):
            keys_to_reset = ['analysis_results', 'semantic_results', 'structure_results', 'competitor_results', 'keyword_results', 'rewriter_results', 'backlink_results', 'copilot_history']
            for key in keys_to_reset:
                if key in st.session_state: del st.session_state[key]
            init_session_state()
            if not url_input and not html_input: st.warning("Please enter a URL or paste HTML to analyze.", icon="⚠️"); return
            auditor = SEOAuditor(url=url_input) if url_input else SEOAuditor(html_content=html_input); st.session_state.analysis_results = auditor.run_audit()
            if not st.session_state.analysis_results: st.error("Core SEO audit failed.", icon="❌"); return
            if st.session_state.api_key_configured:
                ai_model = genai.GenerativeModel('gemini-1.5-flash'); target_text = st.session_state.analysis_results['content']['raw_text']; target_keywords = [k.strip() for k in target_keywords_input.split(',') if k.strip()]
                with st.spinner("Performing AI Analyses..."): st.session_state.semantic_results = perform_semantic_analysis(target_text, ai_model, target_keywords); st.session_state.structure_results = generate_structure_suggestions(target_text, ai_model)
            if url_input:
                competitor_urls = [u.strip() for u in competitor_urls_input.split('\n') if u.strip()][:3]; all_urls_to_compare = [url_input]
                if target_keywords_input and len(competitor_urls) < 3: serp_urls = get_serp_competitor_urls(target_keywords_input.split(',')[0], num_results=2); all_urls_to_compare.extend(serp_urls)
                all_urls_to_compare.extend(list(dict.fromkeys(competitor_urls)))
                if len(all_urls_to_compare) > 1:
                    all_competitor_data = [];
                    with st.spinner(f"Analyzing {len(all_urls_to_compare)-1} competitor(s)..."):
                        for comp_url in all_urls_to_compare[1:]:
                            comp_audit_result = SEOAuditor(url=comp_url).run_audit()
                            if comp_audit_result: all_competitor_data.append({"URL": comp_url.split('//')[-1].split('/')[0], "SEO Score": comp_audit_result['seo_score'],"Word Count": comp_audit_result['content']['word_count'],"Content Text": comp_audit_result['content']['raw_text']})
                    st.session_state.competitor_results = all_competitor_data
            st.session_state.page = "📊 Dashboard"; st.rerun()
def dashboard_page():
    if not st.session_state.get('analysis_results'):
        st.warning("Please run an analysis from the Home page first.", icon="ℹ️")
        if st.button("⬅️ Back to Home"): st.session_state.page = "🏠 Home"; st.rerun()
        return

    res = st.session_state.analysis_results
    st.title("📊 SEO Analysis Dashboard")
    st.markdown(f"**Analysis for:** `{res.get('url', 'N/A')}`")

    # SEO Score Card
    with st.container(border=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            score = res['seo_score']
            st.metric("Overall SEO Score", f"{score}/100")
            if score >= 80: color = "green"
            elif score >= 50: color = "orange"
            else: color = "red"
            st.progress(score, text=f"Score: {score}")


        with col2:
            if res['seo_score'] >= 80:
                st.success("Excellent! This page is well-optimized. Focus on advanced strategies and content quality.", icon="🎉")
            elif res['seo_score'] >= 50:
                st.warning("Good, but there's room for improvement. Address the highlighted issues to boost performance.", icon="📈")
            else:
                st.error("Poor SEO. Significant improvements are needed. Start with the 'Needs Improvement' items below.", icon="🛠️")

    tab1, tab2, tab3, tab4 = st.tabs(["🔍 On-Page", "📝 Content", "⚙️ Technical", "⚡ Performance"])

    with tab1:
        st.subheader("On-Page SEO Elements")
        on_page_cols = st.columns(2)
        with on_page_cols[0]:
            with st.container(border=True):
                st.markdown("**Title Tag**")
                display_check("Presence", res['on_page']['title']['text'] != "Not Found", "A title tag is crucial for SEO.")
                display_check("Length", res['on_page']['title']['status'], f"Current: {res['on_page']['title']['length']} chars (ideal: {CONFIG['seo_scores']['title_range'][0]}-{CONFIG['seo_scores']['title_range'][1]})")
                with st.expander("View Title"): st.code(res['on_page']['title']['text'], language='text')
        with on_page_cols[1]:
            with st.container(border=True):
                st.markdown("**Meta Description**")
                display_check("Presence", res['on_page']['description']['text'] != "Not Found", "A meta description attracts clicks from SERPs.")
                display_check("Length", res['on_page']['description']['status'], f"Current: {res['on_page']['description']['length']} chars (ideal: {CONFIG['seo_scores']['desc_range'][0]}-{CONFIG['seo_scores']['desc_range'][1]})")
                with st.expander("View Description"): st.code(res['on_page']['description']['text'], language='text')
        st.divider()
        st.markdown("**Headings Structure**")
        h1_count = len(res['on_page']['headings']['h1'])
        display_check("H1 Count", h1_count == 1, f"Found {h1_count} H1 tags (should be exactly 1).")
        headings_df = pd.DataFrame([{'Level': f'H{i}', 'Count': len(res['on_page']['headings'][f'h{i}'])} for i in range(1, 7)])
        fig = px.bar(headings_df, x='Level', y='Count', title='Heading Tag Distribution')
        st.plotly_chart(fig, use_container_width=True)


    with tab2:
        st.subheader("Content Analysis")
        content_cols = st.columns(2)
        with content_cols[0]:
            with st.container(border=True):
                st.metric("Word Count", res['content']['word_count'])
                display_check("Minimum Length", res['content']['word_count'] >= CONFIG['seo_scores']['min_word_count'], f"Recommended: >{CONFIG['seo_scores']['min_word_count']} words.")
            with st.container(border=True):
                st.metric("Readability Score (Flesch)", f"{res['content']['readability_score']:.1f}")
                display_check("Readability", res['content']['readability_score'] >= CONFIG['seo_scores']['min_readability'], f"Aim for a score of {CONFIG['seo_scores']['min_readability']}+ for general audiences.")
        with content_cols[1]:
            with st.container(border=True):
                st.metric("Images Found", f"{res['content']['images']['total']}")
                display_check("Alt Text Coverage", res['content']['images']['missing_alt'] == 0, f"{res['content']['images']['missing_alt']} image(s) missing alt text.")
        st.divider()
        st.markdown("**Top Keywords**")
        if res['content']['keywords']:
            keywords_df = pd.DataFrame(res['content']['keywords'], columns=['Keyword', 'Frequency'])
            st.dataframe(keywords_df.head(10), use_container_width=True, hide_index=True)
        else:
            st.info("No significant keywords were extracted from the content.")

    with tab3:
        st.subheader("Technical SEO Checks")
        tech_cols = st.columns(3)
        with tech_cols[0]:
            with st.container(border=True):
                st.markdown("**Core Vitals**")
                display_check("HTTPS", res['technical']['https'], "Secure connection is a positive ranking signal.")
                display_check("Mobile Friendly", res['technical']['mobile_friendly_meta'], "Viewport meta tag is configured for mobile devices.")
                display_check("Canonical Tag", res['technical']['canonical'], "Helps prevent duplicate content issues.")
        with tech_cols[1]:
            with st.container(border=True):
                st.markdown("**Crawlability**")
                display_check("Robots.txt", res['technical']['robots_txt_present'], "A robots.txt file was found on the domain.")
                display_check("Sitemap.xml", res['technical']['sitemap_xml_present'], "A sitemap.xml file was found on the domain.")
                display_check("Internal Links", res['technical']['internal_links_count'] >= CONFIG['seo_scores']['min_internal_links'], f"Found {res['technical']['internal_links_count']} (min {CONFIG['seo_scores']['min_internal_links']} recommended).")
        with tech_cols[2]:
            with st.container(border=True):
                st.markdown("**Structured Data**")
                display_check("Schema Markup", res['technical']['schema_present'], "Schema (e.g., JSON-LD) can enable rich snippets.")

    with tab4:
        st.subheader("Performance Metrics")
        perf_cols = st.columns(2)
        with perf_cols[0]:
            with st.container(border=True):
                if isinstance(res['performance']['load_time'], float):
                    load_time = res['performance']['load_time']
                    st.metric("Page Load Time", f"{load_time:.2f}s")
                    display_check("Load Time", load_time < CONFIG['seo_scores']['max_load_time'], f"Should be under {CONFIG['seo_scores']['max_load_time']}s for good user experience.")
                else:
                    st.metric("Page Load Time", "N/A")
        with perf_cols[1]:
            with st.container(border=True):
                st.metric("Page Size", f"{res['performance']['size_kb']:.1f} KB")

def keyword_research_page():
    st.title("🔑 Keyword Research")
    if not st.session_state.api_key_configured: st.warning("This tool requires a Gemini API Key configured in your secrets.", icon="⚠️"); return

    with st.form("keyword_form"):
        seed_keyword = st.text_input("Enter a seed keyword", placeholder="e.g., 'digital marketing'")
        submitted = st.form_submit_button("Generate Keywords", type="primary", use_container_width=True)

        if submitted and seed_keyword:
            with st.spinner("AI is performing keyword research..."):
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    prompt = f"""
                    Act as a keyword research expert. For the seed keyword '{seed_keyword}', generate a comprehensive analysis.
                    Present the output as a single, valid JSON object with this structure:
                    {{
                        "related_keywords": [
                            {{"keyword": "...", "intent": "Informational/Commercial/...", "estimated_volume": "Low/Medium/High", "difficulty": "Low/Medium/High"}}
                        ],
                        "long_tail_questions": [],
                        "content_ideas": {{
                            "blog_posts": [],
                            "guides": [],
                            "tools": []
                        }},
                        "competitor_angle": []
                    }}
                    Provide 10 related keywords, 5 long-tail questions, 3 ideas for each content type, and 3 competitor angles.
                    """
                    response = model.generate_content(prompt)
                    json_response = response.text.strip().lstrip("```json").rstrip("```")
                    st.session_state.keyword_results = json.loads(json_response)
                except Exception as e:
                    st.error(f"Failed to generate keywords: {e}")

    if st.session_state.get('keyword_results'):
        res = st.session_state.keyword_results
        st.subheader("📈 Related Keywords & Intent")
        keywords_df = pd.DataFrame(res.get('related_keywords', []))
        if not keywords_df.empty:
            st.dataframe(keywords_df, hide_index=True, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("❓ Long-Tail Questions")
            for question in res.get('long_tail_questions', []): st.markdown(f"- {question}")
        with col2:
            st.subheader("🏆 Competitor Angle")
            for tip in res.get('competitor_angle', []): st.markdown(f"- {tip}")

        st.subheader("💡 Content Ideas")
        tabs = st.tabs(res.get('content_ideas', {}).keys())
        for i, (ctype, ideas) in enumerate(res.get('content_ideas', {}).items()):
            with tabs[i]:
                for idea in ideas: st.markdown(f"- {idea}")
def content_rewriter_page():
    st.title("✍️ AI Content Rewriter & Optimizer")
    if not st.session_state.api_key_configured: st.warning("This tool requires a Gemini API Key.", icon="⚠️"); return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Your Original Text")
        text_to_rewrite = st.text_area("Paste your text here to rewrite", height=300, label_visibility="collapsed")
        with st.form("rewriter_form"):
            style = st.selectbox("Choose a rewriting style", ["Professional", "Conversational", "Benefit-Focused", "Concise", "Expanded", "SEO-Optimized"])
            tone = st.selectbox("Choose a tone", ["Neutral", "Friendly", "Authoritative", "Persuasive", "Casual"])
            submitted = st.form_submit_button("Rewrite Content", type="primary", use_container_width=True)

            if submitted and text_to_rewrite:
                with st.spinner(f"AI is rewriting in a {style} style with a {tone} tone..."):
                    try:
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        prompt = f"""
                        Rewrite the following text in 3 different variations based on these parameters:
                        - Style: {style}
                        - Tone: {tone}
                        - Goal: Each variation must maintain the core message but offer a unique angle, structure, or vocabulary.
                        - For 'SEO-Optimized' style, please include relevant keywords naturally.

                        Present the output as a valid JSON object with the structure:
                        {{
                            "variations": [
                                {{"title": "Variation 1: [Brief Name]", "text": "...", "explanation": "..."}}
                            ]
                        }}
                        Original Text:
                        {text_to_rewrite[:4000]}
                        """
                        response = model.generate_content(prompt)
                        json_response = response.text.strip().lstrip("```json").rstrip("```")
                        st.session_state.rewriter_results = json.loads(json_response)
                    except Exception as e:
                        st.error(f"Failed to rewrite content: {e}")

    with col2:
        st.subheader("✨ Rewritten Variations")
        if not st.session_state.get('rewriter_results'):
            st.info("Your rewritten content will appear here.")
        else:
            for i, variation in enumerate(st.session_state.rewriter_results.get('variations', [])):
                with st.expander(f"**{variation.get('title', f'Variation {i+1}')}**", expanded=i==0):
                    st.markdown(f"***Explanation:*** *{variation.get('explanation', '')}*")
                    st.markdown("---")
                    st.markdown(variation.get('text', ''))

# --- ENHANCED BACKLINK VERIFIER ---
def _verify_single_url(referring_url: str, target_domain: str) -> List[Dict]:
    """Performs a multi-step verification for a single referring URL, finding all backlinks on the page."""
    headers = {'User-Agent': CONFIG['api']['user_agent']}
    page_info = {"Referring URL": referring_url, "Page Status": "Error", "HTTP Code": None, "Final URL (after redirects)": referring_url, "Page Title": "N/A"}
    try:
        response = requests.get(referring_url, headers=headers, timeout=CONFIG['api']['timeout'], allow_redirects=True)
        page_info.update({"HTTP Code": response.status_code, "Final URL (after redirects)": response.url})
        if not response.ok:
            return [{**page_info, "Link Found (in <a> tag)": "N/A", "Anchor Text": "N/A", "Is Nofollow": "N/A", "Link URL": "N/A"}]

        page_info["Page Status"] = "Live"
        soup = BeautifulSoup(response.content, 'html.parser')
        title_tag = soup.find('title')
        page_info["Page Title"] = title_tag.string.strip() if title_tag else "Not Found"

        found_links = []
        for a_tag in soup.find_all('a', href=True):
            abs_href = urljoin(response.url, a_tag['href'])
            if urlparse(abs_href).netloc == target_domain:
                link_details = {
                    "Link Found (in <a> tag)": "Yes",
                    "Anchor Text": a_tag.get_text(strip=True) or "[No Anchor Text]",
                    "Is Nofollow": "Yes" if 'nofollow' in a_tag.get('rel', []) else "No",
                    "Link URL": abs_href
                }
                found_links.append({**page_info, **link_details})

        if not found_links:
            return [{**page_info, "Link Found (in <a> tag)": "No", "Anchor Text": "N/A", "Is Nofollow": "N/A", "Link URL": "N/A"}]
        return found_links
    except requests.exceptions.RequestException as e:
        page_info["Page Status"] = "Request Error"
        return [{**page_info, "Link Found (in <a> tag)": "N/A", "Anchor Text": "N/A", "Is Nofollow": "N/A", "Link URL": "N/A"}]

def _display_backlink_summary(df: pd.DataFrame, target_domain: str):
    """Calculates and displays summary stats and anchor text analysis."""
    st.subheader("📊 Summary & Statistics")
    links_found_df = df[df['Link Found (in <a> tag)'] == 'Yes'].copy()
    stats_cols = st.columns(4)
    stats_cols[0].metric("Total Backlinks Found", len(links_found_df))
    stats_cols[1].metric("Unique Linking Pages", links_found_df['Referring URL'].nunique())
    stats_cols[2].metric("DoFollow Links", len(links_found_df[links_found_df['Is Nofollow'] == 'No']))
    stats_cols[3].metric("Unique Anchor Texts", links_found_df['Anchor Text'].nunique())

    if links_found_df.empty: st.info("No backlinks were found to analyze anchor text."); return
    st.divider()
    st.subheader("⚓ Anchor Text Analysis")
    analysis_cols = st.columns(2)
    with analysis_cols[0]:
        st.markdown("**Top 10 Anchor Texts**")
        anchor_counts = links_found_df['Anchor Text'].value_counts().reset_index()
        anchor_counts.columns = ['Anchor Text', 'Count']
        fig = px.bar(anchor_counts.head(10), x='Anchor Text', y='Count', text_auto=True)
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    with analysis_cols[1]:
        st.markdown("**Anchor Text Distribution**")
        brand_name = target_domain.split('.')[0]
        generic_terms = ['click here', 'read more', 'website', 'link', 'more info', 'learn more', 'here']
        def categorize_anchor(row):
            anchor = str(row['Anchor Text']).lower().strip()
            if not anchor or anchor == "[no anchor text]": return "No Text"
            if brand_name in anchor: return "Branded"
            if target_domain in anchor: return "Naked URL"
            if anchor in generic_terms: return "Generic"
            return "Other/Keyword"
        links_found_df['Anchor Type'] = links_found_df.apply(categorize_anchor, axis=1)
        anchor_dist = links_found_df['Anchor Type'].value_counts().reset_index()
        anchor_dist.columns = ['Type', 'Count']
        fig_pie = px.pie(anchor_dist, values='Count', names='Type', title='Distribution by Type', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

def _display_backlink_table(df: pd.DataFrame):
    """Displays filter options and the resulting DataFrame."""
    st.subheader("📋 Detailed Link Report")
    filter_cols = st.columns(3)
    with filter_cols[0]:
        status_filter = st.multiselect("Filter by Page Status", options=df['Page Status'].unique(), default=[s for s in df['Page Status'].unique() if s not in ['Error', 'Request Error']])
    with filter_cols[1]:
        nofollow_filter = st.selectbox("Filter by Link Type", options=["All", "DoFollow Only", "NoFollow Only"], index=0)
    with filter_cols[2]:
        link_found_filter = st.selectbox("Filter by Link Presence", options=["All", "Link Found", "Link Not Found"], index=0)

    filtered_df = df.copy()
    if status_filter: filtered_df = filtered_df[filtered_df['Page Status'].isin(status_filter)]
    if nofollow_filter == "DoFollow Only": filtered_df = filtered_df[filtered_df['Is Nofollow'] == 'No']
    elif nofollow_filter == "NoFollow Only": filtered_df = filtered_df[filtered_df['Is Nofollow'] == 'Yes']
    if link_found_filter == "Link Found": filtered_df = filtered_df[filtered_df['Link Found (in <a> tag)'] == 'Yes']
    elif link_found_filter == "Link Not Found": filtered_df = filtered_df[filtered_df['Link Found (in <a> tag)'] == 'No']

    st.dataframe(filtered_df, hide_index=True, use_container_width=True,
        column_config={ "Referring URL": st.column_config.LinkColumn("Referring URL"), "Final URL (after redirects)": st.column_config.LinkColumn("Final URL"), "Link URL": st.column_config.LinkColumn("Backlink URL"),})
    st.download_button(label="📥 Download Full Report as CSV", data=df.to_csv(index=False).encode('utf-8'), file_name=f"backlink_report_{df['Referring URL'].nunique()}_pages.csv", mime="text/csv", use_container_width=True)

def backlink_verifier_page():
    st.title("🔗 Backlink Verifier")
    st.info("This tool verifies if a list of pages contains backlinks to your target domain.", icon="ℹ️")

    with st.form("verifier_form"):
        target_domain_input = st.text_input("Your Target Domain", placeholder="yourwebsite.com", help="Enter just the domain name, e.g., 'google.com'")
        referring_urls_input = st.text_area("Enter Referring URLs to Verify (one per line)", height=200, placeholder="https://example.com/blog-post\nhttps://another-site.net/article")
        submitted = st.form_submit_button("🔍 Verify Backlinks", type="primary", use_container_width=True)

    if submitted:
        target_domain = target_domain_input.strip().replace("https://", "").replace("http://", "").split("/")[0]
        referring_urls = [url.strip() for url in referring_urls_input.split('\n') if url.strip()]

        if not target_domain or not referring_urls:
            st.warning("Please provide a target domain and at least one referring URL.", icon="⚠️"); return

        if len(referring_urls) > 20:
            st.warning(f"For performance, only the first 20 of {len(referring_urls)} URLs will be processed.", icon="⚙️")
            referring_urls = referring_urls[:20]

        all_results = []
        progress_bar = st.progress(0, text="Starting verification...")
        for i, url in enumerate(referring_urls):
            progress_bar.progress((i + 1) / len(referring_urls), text=f"Verifying {i+1}/{len(referring_urls)}: {url[:60]}...")
            url_results = _verify_single_url(url, target_domain)
            all_results.extend(url_results)
        progress_bar.empty()

        st.session_state.backlink_results = all_results
        st.session_state.backlink_target_domain = target_domain
        st.rerun()

    if 'backlink_results' in st.session_state and st.session_state.backlink_results is not None:
        df = pd.DataFrame(st.session_state.backlink_results)
        target_domain = st.session_state.get('backlink_target_domain', "")
        links_found_count = len(df[df['Link Found (in <a> tag)'] == 'Yes'])
        pages_checked_count = df['Referring URL'].nunique()

        st.success(f"Verification Complete! Found **{links_found_count}** backlink(s) across **{pages_checked_count}** checked page(s).")
        st.divider()

        _display_backlink_summary(df, target_domain)
        st.divider()
        _display_backlink_table(df)

def ai_copilot_page():
    st.title("🤖 AI SEO Copilot")
    if not st.session_state.api_key_configured: st.warning("This tool requires a Gemini API Key.", icon="⚠️"); return

    if 'copilot_history' not in st.session_state: st.session_state.copilot_history = []
    for msg in st.session_state.copilot_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Ask for SEO advice based on the current analysis..."):
        st.session_state.copilot_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    context_parts = []
                    if st.session_state.get('analysis_results'): context_parts.append(f"SEO Audit Results:\n{json.dumps(st.session_state.analysis_results, indent=2, default=str)}")
                    if st.session_state.get('semantic_results'): context_parts.append(f"Semantic Analysis:\n{json.dumps(st.session_state.semantic_results, indent=2, default=str)}")
                    if st.session_state.get('keyword_results'): context_parts.append(f"Keyword Research:\n{json.dumps(st.session_state.keyword_results, indent=2, default=str)}")
                    if st.session_state.get('backlink_results'): context_parts.append(f"Backlink Data (first 3 items):\n{json.dumps(st.session_state.backlink_results[:3], indent=2, default=str)}")
                    context = "\n\n".join(context_parts) if context_parts else "No specific audit data available."
                    full_prompt = f"""You are an expert SEO analyst assistant. Provide detailed, actionable advice based on the available context and the user's question.
                    Available Context: {context[:10000]}
                    User Question: {prompt}
                    Guidelines: Be specific, use markdown, prioritize recommendations, and explain technical terms simply."""
                    response = model.generate_content(full_prompt)
                    ai_response = response.text
                    st.markdown(ai_response)
                    st.session_state.copilot_history.append({"role": "assistant", "content": ai_response})
                except Exception as e:
                    st.error(f"Could not generate response from AI: {e}")

# --- MAIN APPLICATION RUNNER ---
def main():
    load_css()
    if 'api_key_configured' not in st.session_state:
        st.session_state.api_key_configured = configure_ai()
    init_session_state()

    with st.sidebar:
        st.header(f"{CONFIG['app']['icon']} {CONFIG['app']['title']}")
        PAGES = {
            "🏠 Home": home_page,
            "📊 Dashboard": dashboard_page,
            "--- Tools ---": None,
            "🔑 Keyword Research": keyword_research_page,
            "✍️ Content Rewriter": content_rewriter_page,
            "🔗 Backlink Verifier": backlink_verifier_page,
            "🤖 AI SEO Copilot": ai_copilot_page,
        }

        for page_name, page_func in PAGES.items():
            if page_func is None:
                st.markdown(f"**{page_name}**")
                continue

            is_active = (st.session_state.page == page_name)
            disable_for_ai = page_name in ["🔑 Keyword Research", "✍️ Content Rewriter", "🤖 AI SEO Copilot"] and not st.session_state.api_key_configured
            disable_for_results = page_name in ["📊 Dashboard"] and not st.session_state.analysis_results

            if st.button(page_name, use_container_width=True, key=f"nav_{page_name}", disabled=(disable_for_ai or disable_for_results), type="primary" if is_active else "secondary"):
                st.session_state.page = page_name
                st.rerun()

        st.markdown("---")
        st.info(f"App Version: {CONFIG['app']['version']}")
        st.markdown("[GitHub](https://github.com/your-repo-link) | [Feedback](mailto:your-email@example.com)", unsafe_allow_html=True)

    page_to_render = PAGES.get(st.session_state.page, home_page)
    if page_to_render:
        page_to_render()

if __name__ == "__main__":
    main()
# --- IMPORTS ---
import os
import time
import json
import requests
import warnings
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter
from urllib.parse import urlparse, urljoin, quote_plus

import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import textstat
import extruct
from w3lib.html import get_base_url
from bs4 import BeautifulSoup

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
        "version": "14.3", # Version bump for prompt fix
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
        "timeout": 30
    }
}

# --- INITIALIZATION & SETUP ---

@st.cache_resource
def setup_nltk():
    """Downloads all necessary NLTK data models if not already present."""
    resources = [
        ("punkt", "tokenizers/punkt"),
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger")
    ]
    for resource_id, path in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource_id, quiet=True)
    return True

@st.cache_resource
def configure_ai():
    """Configures the Gemini API key and tests the connection."""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            # Test a small call to ensure API is working by using GenerativeModel
            model = genai.GenerativeModel('gemini-1.5-flash')
            model.generate_content("test", request_options={'timeout': 5})
            return True
        except Exception as e:
            st.error(f"Failed to configure AI or connect to Gemini API. Please check your `GEMINI_API_KEY` in `secrets.toml`. Error: {e}", icon="🔑")
            return False
    st.warning("Gemini API Key not found. Some AI-powered features will be disabled. Please add `GEMINI_API_KEY = 'YOUR_API_KEY'` to your `.streamlit/secrets.toml` file.", icon="🔑")
    return False

def init_session_state():
    """Initializes session state variables."""
    defaults = {
        'analysis_results': None,
        'page': "🏠 Home",
        'copilot_history': [],
        'keyword_results': None,
        'rewriter_results': None,
        'backlink_results': None,
        'semantic_results': None,
        'structure_results': None,
        'competitor_results': None,
        'api_key_configured': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- CORE LOGIC: SEO AUDITOR CLASS ---

class SEOAuditor:
    def __init__(self, url: Optional[str] = None, html_content: Optional[str] = None):
        self.url = url
        self.html_content = html_content
        self.results: Dict[str, Any] = {}
        self.soup: Optional[BeautifulSoup] = None
        self.headers = {'User-Agent': CONFIG['api']['user_agent']}
        self.perf_data = {}
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words_set = set(stopwords.words('english'))

    def run_audit(self) -> Optional[Dict[str, Any]]:
        try:
            if not self._fetch_and_parse_html():
                return None
            
            # Re-initialize results to ensure clean state for each run
            self.results = {
                'performance': self.perf_data,
                'on_page': self._check_on_page(),
                'content': self._analyze_content(),
                'technical': self._check_technical(),
                'url': self.url
            }
            self.results['seo_score'] = self._calculate_seo_score()
            return self.results
        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {e}", icon="🚨")
            return None

    def _fetch_and_parse_html(self) -> bool:
        try:
            if self.url:
                if not (self.url.startswith('http://') or self.url.startswith('https://')):
                    raise ValueError("Invalid URL. Please include http:// or https://")
                start_time = time.time()
                response = requests.get(self.url, headers=self.headers, timeout=CONFIG['api']['timeout'])
                response.raise_for_status()
                self.html_content = response.text
                self.perf_data = {'load_time': time.time() - start_time, 'size_kb': len(response.content) / 1024}
            elif self.html_content:
                self.url = "Pasted Content"
                self.perf_data = {'load_time': 'N/A', 'size_kb': len(self.html_content.encode('utf-8')) / 1024}
            else:
                raise ValueError("Either a URL or HTML content must be provided.")
            
            self.soup = BeautifulSoup(self.html_content, 'html.parser')
            # Remove scripts, styles, and navigational/structural elements for cleaner text extraction
            for element in self.soup(["script", "style", "noscript", "iframe", "header", "footer", "nav", "aside", "form"]):
                element.decompose()
            return True
        except requests.exceptions.RequestException as e:
            st.error(f"Network Error: Could not retrieve URL. {e}", icon="🌐")
            return False
        except ValueError as e:
            st.warning(str(e), icon="⚠️")
            return False

    def _check_on_page(self) -> Dict[str, Any]:
        title_tag = self.soup.find('title')
        title_text = title_tag.string.strip() if title_tag and title_tag.string else "Not Found"
        desc_tag = self.soup.find('meta', attrs={'name': 'description'})
        desc_text = desc_tag.get('content', 'Not Found').strip() if desc_tag else "Not Found"
        
        # Check for multiple H1s
        h1_tags = self.soup.find_all('h1')
        
        return {
            'title': {'text': title_text, 'length': len(title_text), 'status': CONFIG['seo_scores']['title_range'][0] <= len(title_text) <= CONFIG['seo_scores']['title_range'][1]},
            'description': {'text': desc_text, 'length': len(desc_text), 'status': CONFIG['seo_scores']['desc_range'][0] <= len(desc_text) <= CONFIG['seo_scores']['desc_range'][1]},
            'headings': {f'h{i}': [h.get_text(strip=True) for h in self.soup.find_all(f'h{i}')] for i in range(1, 7)},
            'multiple_h1': len(h1_tags) > 1,
            'h1_present': len(h1_tags) > 0,
            'image_alt_status': self._check_image_alt_text()
        }

    def _analyze_content(self) -> Dict[str, Any]:
        main_content_area = self.soup.find('main') or self.soup.find('article') or self.soup.body
        text = main_content_area.get_text(separator=' ', strip=True) if main_content_area else ""
        
        words = [self.lemmatizer.lemmatize(word) for word in word_tokenize(text.lower()) if word.isalpha()]
        filtered_words = [word for word in words if word not in self.stop_words_set]
        
        images = self.soup.find_all('img')
        total_img, alt_img = len(images), sum(1 for img in images if img.get('alt', '').strip())
        
        return {
            'raw_text': text,
            'word_count': len(words),
            'readability_score': textstat.flesch_reading_ease(text) if text else 0,
            'keywords': Counter(filtered_words).most_common(20),
            'images': {'total': total_img, 'with_alt': alt_img, 'missing_alt': total_img - alt_img}
        }

    def _check_image_alt_text(self) -> str:
        images = self.soup.find_all('img')
        missing_alt_count = sum(1 for img in images if not img.get('alt', '').strip())
        total_images = len(images)
        
        if total_images == 0:
            return "No images found."
        elif missing_alt_count == 0:
            return "All images have alt text. Great!"
        else:
            return f"{missing_alt_count} of {total_images} images are missing alt text."

    def _check_technical(self) -> Dict[str, Any]:
        base_url = get_base_url(self.html_content, self.url) if self.url != "Pasted Content" else ""
        domain = urlparse(base_url).netloc
        internal_links_count = 0
        
        if domain:
            for link in self.soup.find_all('a', href=True):
                full_link = urljoin(base_url, link['href'])
                if full_link.startswith('http://') or full_link.startswith('https://'):
                    if domain in urlparse(full_link).netloc:
                        internal_links_count += 1
        
        robots_txt_present = False
        sitemap_xml_present = False
        if self.url and self.url != "Pasted Content":
            root_domain = f"{urlparse(self.url).scheme}://{urlparse(self.url).netloc}"
            try:
                robots_response = requests.get(urljoin(root_domain, '/robots.txt'), headers=self.headers, timeout=5)
                if robots_response.status_code == 200 and "user-agent" in robots_response.text.lower():
                    robots_txt_present = True
            except requests.exceptions.RequestException:
                pass

            try:
                sitemap_response = requests.get(urljoin(root_domain, '/sitemap.xml'), headers=self.headers, timeout=5)
                if sitemap_response.status_code == 200 and "<urlset" in sitemap_response.text.lower():
                    sitemap_xml_present = True
            except requests.exceptions.RequestException:
                pass

        return {
            'https': self.url.startswith('https://') if self.url != "Pasted Content" else False,
            'canonical': bool(self.soup.find('link', rel='canonical')),
            'schema_present': bool(extruct.extract(self.html_content, syntaxes=['json-ld'])),
            'viewport': bool(self.soup.find('meta', attrs={'name': 'viewport'})),
            'internal_links_count': internal_links_count,
            'robots_txt_present': robots_txt_present,
            'sitemap_xml_present': sitemap_xml_present,
            'mobile_friendly_meta': bool(self.soup.find('meta', attrs={'name': 'viewport', 'content': True}) and ('width=device-width' in self.soup.find('meta', attrs={'name': 'viewport'})['content'])),
        }

    def _calculate_seo_score(self) -> int:
        score, weights = 0, CONFIG['seo_scores']['weights']
        total_weight, res, s_conf = sum(weights.values()), self.results, CONFIG['seo_scores']
        checks = {
            'title_ok': res['on_page']['title']['status'],
            'desc_ok': res['on_page']['description']['status'],
            'h1_ok': len(res['on_page']['headings']['h1']) == 1,
            'https': res['technical']['https'],
            'content_long': res['content']['word_count'] >= s_conf['min_word_count'],
            'alt_tags_ok': res['content']['images']['total'] == 0 or (res['content']['images']['with_alt'] / max(1, res['content']['images']['total'])) >= 0.9,
            'readability_ok': res['content']['readability_score'] >= s_conf['min_readability'],
            'canonical': res['technical']['canonical'],
            'schema': res['technical']['schema_present'],
            'viewport': res['technical']['viewport'],
            'internal_links': res['technical'].get('internal_links_count', 0) >= s_conf['min_internal_links'],
            'load_time_ok': isinstance(res['performance']['load_time'], float) and res['performance']['load_time'] < s_conf['max_load_time']
        }
        for check, is_ok in checks.items():
            if is_ok:
                score += weights.get(check, 0)
        return int((score / total_weight) * 100) if total_weight > 0 else 0

# --- NEW FEATURE IMPLEMENTATIONS ---

def extract_text_from_url(url: str) -> Optional[str]:
    """Helper function to scrape text content from a URL."""
    try:
        headers = {'User-Agent': CONFIG['api']['user_agent']}
        response = requests.get(url, headers=headers, timeout=CONFIG['api']['timeout'])
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav", "aside", "form"]):
            element.decompose()
        main_content_area = soup.find('main') or soup.find('article') or soup.body
        return main_content_area.get_text(separator=' ', strip=True) if main_content_area else ""
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not scrape content from {url}: {e}")
        return None
    except Exception as e:
        st.warning(f"Error extracting text from {url}: {e}")
        return None

def perform_semantic_analysis(text: str, model, target_keywords: List[str] = None) -> Dict[str, Any]:
    """Performs semantic analysis using an LLM."""
    if not text:
        return {"summary": "No content to analyze.", "entities": [], "lsi_keywords": [], "readability_score": 0}

    readability_score = textstat.flesch_reading_ease(text)

    prompt_keywords_section = ""
    if target_keywords:
        prompt_keywords_section = f"4. Additionally, check if the following target keywords are covered well in the text: {', '.join(target_keywords)}. Comment on their presence and semantic relevance."

    # Conditionally create the list of keys for the JSON output instruction
    keys_for_json = "'summary', 'entities', 'lsi_keywords'"
    if target_keywords:
        keys_for_json += ", 'keyword_coverage'"

    prompt = f"""
    As an expert SEO analyst, analyze the following text for semantic SEO. Provide:
    1. A brief summary of the text's core topic.
    2. A list of key entities (people, places, organizations, concepts) mentioned.
    3. A list of suggested LSI (Latent Semantic Indexing) keywords that are contextually related to the text, aiming for 5-10 distinct terms.
    {prompt_keywords_section}
    
    Present the output as a single, valid JSON object with the following keys ONLY: {keys_for_json}.
    
    Text to analyze:
    {text[:4000]}
    """
    try:
        response = model.generate_content(prompt)
        json_response = response.text.strip()
        if json_response.startswith("```json") and json_response.endswith("```"):
            json_response = json_response[len("```json"):-len("```")]
        
        parsed_results = json.loads(json_response)
        parsed_results['readability_score'] = readability_score
        return parsed_results
    except json.JSONDecodeError as e:
        st.error(f"AI response was not valid JSON for semantic analysis: {e}. Raw response: {response.text}")
        return {"summary": "Error parsing AI response.", "entities": [], "lsi_keywords": [], "readability_score": readability_score}
    except Exception as e:
        st.error(f"Failed to perform semantic analysis with AI: {e}")
        return {"summary": "AI analysis failed.", "entities": [], "lsi_keywords": [], "readability_score": readability_score}

def generate_structure_suggestions(text: str, model) -> str:
    """Generates structure optimization suggestions using LLM."""
    if not text:
        return "No content provided for structure optimization."

    prompt = f"""
    As a content editor specializing in on-page SEO, analyze the structure of the following text.
    Provide a rewritten version of the text with an optimized structure. Your suggestions should focus on:
    - Improving the heading hierarchy (H1, H2, H3). If an H1 is missing, suggest one. Ensure only one H1.
    - Breaking down long paragraphs into shorter, more readable ones.
    - Using bullet points or numbered lists for listable content where appropriate.
    - Bolding key phrases for emphasis and scannability.
    - Ensuring a logical flow of information.
    
    Return only the optimized markdown text, suitable for direct display. Do not include any preambles or explanations outside the markdown.
    
    Text to optimize:
    {text[:5000]}
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Failed to generate structure suggestions with AI: {e}")
        return "AI structure optimization failed."

def get_serp_competitor_urls(keyword: str, num_results: int = 5) -> List[str]:
    """
    Scrapes Google SERP for a given keyword to find top competitor URLs.
    NOTE: Google often blocks automated scraping. This is a best-effort attempt.
    """
    search_url = f"https://www.google.com/search?q={quote_plus(keyword)}"
    headers = {'User-Agent': CONFIG['api']['user_agent']}
    competitor_urls = []
    try:
        response = requests.get(search_url, headers=headers, timeout=CONFIG['api']['timeout'])
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('http') and "google.com" not in href and "/url?q=" in href:
                actual_url = href.split("/url?q=")[1].split("&")[0]
                if urlparse(actual_url).netloc and actual_url not in competitor_urls:
                    competitor_urls.append(actual_url)
                    if len(competitor_urls) >= num_results:
                        break
        if not competitor_urls:
            st.warning(f"Could not find organic search results for '{keyword}'. Google might have blocked the request or the HTML structure has changed. Consider manually adding competitor URLs.")
        return competitor_urls

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to scrape SERP for '{keyword}': {e}. Google often blocks automated requests. Please try again later or provide competitor URLs manually.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred during SERP scraping: {e}")
        return []

def get_content_similarity(text1: str, text2: str) -> float:
    """Calculates cosine similarity between two texts using TF-IDF."""
    if not text1 or not text2:
        return 0.0

    lemmatizer = WordNetLemmatizer()
    def custom_tokenizer(text):
        return [lemmatizer.lemmatize(w) for w in word_tokenize(text) if w.isalpha()]

    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True,
                                 tokenizer=custom_tokenizer)
    
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(f"{similarity:.2f}")
    except Exception as e:
        st.warning(f"Could not calculate content similarity: {e}. Returning 0.0.")
        return 0.0

# --- UI & PAGE RENDERING ---

def load_css():
    theme = CONFIG['theme']
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, [class*="st-"] {{ font-family: 'Inter', sans-serif; color: {theme['text_color']}; }}
        .stApp {{ background-color: {theme['bg_color']}; }}
        div[data-testid="stSidebar"] > div:first-child {{ background-color: {theme['card_bg']}; }}
        .stButton>button {{ border-radius: 0.5rem; }}
        .st-emotion-cache-1y4p8pa {{
            background: linear-gradient(135deg, {theme['primary_color']} 0%, {theme['accent_color']} 100%);
            color: white; border: none;
        }}
    </style>""", unsafe_allow_html=True)

def display_check(label: str, is_ok: bool):
    status_span = f'<span style="color: {"#16a34a" if is_ok else "#dc2626"};">{"✅ OK" if is_ok else "❌ Needs Improvement"}</span>'
    st.markdown(f"**{label}:** {status_span}", unsafe_allow_html=True)

def home_page():
    st.title(f"{CONFIG['app']['icon']} {CONFIG['app']['title']}")
    st.markdown("Enter a URL or paste raw HTML for an instant, in-depth SEO audit. You can also specify keywords and competitor URLs for richer analysis.")

    with st.form(key="analysis_form"):
        url_input = st.text_input("Your Website URL", placeholder="https://www.yourdomain.com/your-page-to-analyze")
        html_input = st.text_area("Or Paste Raw HTML (Overrides URL if both are provided)", height=150)
        
        st.markdown("---")
        st.subheader("Optional: Enhance Analysis")
        target_keywords_input = st.text_input("Target Keywords (comma-separated)", placeholder="e.g., 'best SEO tools, SEO analysis software'")
        competitor_urls_input = st.text_area("Competitor URLs (one per line, up to 3)", placeholder="https://www.competitor1.com/page\nhttps://www.competitor2.com/page")

        if st.form_submit_button("Perform All Analyses", type="primary", use_container_width=True):
            st.session_state.analysis_results = None
            st.session_state.semantic_results = None
            st.session_state.structure_results = None
            st.session_state.competitor_results = None

            target_keywords = [k.strip() for k in target_keywords_input.split(',') if k.strip()]
            competitor_urls = [u.strip() for u in competitor_urls_input.split('\n') if u.strip()][:3]

            if not url_input and not html_input:
                st.warning("Please enter a URL or paste HTML to analyze.", icon="⚠️")
                return

            main_auditor = None
            with st.spinner("Step 1/3: Performing Core SEO Audit... 🕵️‍♀️"):
                if html_input: # Prioritize HTML input
                    main_auditor = SEOAuditor(html_content=html_input)
                    st.session_state.analysis_results = main_auditor.run_audit()
                elif url_input:
                    main_auditor = SEOAuditor(url=url_input)
                    st.session_state.analysis_results = main_auditor.run_audit()
                
                if not st.session_state.analysis_results:
                    st.error("Core SEO audit failed. Cannot proceed with further analysis.", icon="❌")
                    return

            if st.session_state.api_key_configured:
                ai_model = genai.GenerativeModel('gemini-1.5-flash')

                with st.spinner("Step 2/3: Performing Semantic Analysis... 🧠"):
                    target_content_text = st.session_state.analysis_results['content']['raw_text']
                    st.session_state.semantic_results = perform_semantic_analysis(target_content_text, ai_model, target_keywords)
                
                with st.spinner("Step 3/3: Generating Structure Optimization Suggestions... 📐"):
                    st.session_state.structure_results = generate_structure_suggestions(target_content_text, ai_model)
            else:
                st.info("AI-powered features (Semantic Analysis, Structure Optimizer) skipped due to missing Gemini API Key.", icon="🔑")

            if url_input:
                all_competitor_data = []
                comparison_urls = [url_input]

                if target_keywords:
                    st.info("Attempting to find top competitors from SERP for provided keywords...", icon="🔍")
                    serp_competitors = []
                    for kw in target_keywords:
                        serp_competitors.extend(get_serp_competitor_urls(kw, num_results=2))
                    for u in serp_competitors:
                        if u not in comparison_urls and len(competitor_urls) < 3:
                            competitor_urls.append(u)

                final_competitor_urls = list(set(competitor_urls))[:3]
                comparison_urls.extend(final_competitor_urls)

                if len(comparison_urls) > 1:
                    with st.spinner(f"Final Step: Analyzing Competitors... 🆚"):
                        progress_text = "Analyzing competitor URLs, this may take a moment..."
                        progress_bar = st.progress(0, text=progress_text)
                        
                        for i, comp_url in enumerate(comparison_urls):
                            try:
                                comp_auditor = SEOAuditor(url=comp_url)
                                comp_audit_result = comp_auditor.run_audit()
                                if comp_audit_result:
                                    all_competitor_data.append({
                                        "URL": comp_url,
                                        "SEO Score": comp_audit_result['seo_score'],
                                        "Word Count": comp_audit_result['content']['word_count'],
                                        "Readability": f"{comp_audit_result['content']['readability_score']:.1f}",
                                        "Load Time (s)": f"{comp_audit_result['performance']['load_time']:.2f}" if isinstance(comp_audit_result['performance']['load_time'], float) else "N/A",
                                        "Page Size (KB)": f"{comp_audit_result['performance']['size_kb']:.1f}",
                                        "Main Content Text": comp_audit_result['content']['raw_text'],
                                    })
                                else:
                                    st.warning(f"Could not perform full audit for competitor {comp_url}.")
                            except Exception as e:
                                st.error(f"Error during competitor analysis for {comp_url}: {e}")
                            progress_bar.progress((i + 1) / len(comparison_urls), text=f"Analyzing {comp_url} ({i+1}/{len(comparison_urls)})")
                        progress_bar.empty()
                        
                        st.session_state.competitor_results = all_competitor_data
                else:
                    st.info("No competitor URLs provided or found to perform competitor analysis.")
            else:
                st.info("Competitor analysis skipped as no target URL was provided for comparison.")

            if st.session_state.analysis_results:
                st.session_state.page = "📊 Dashboard"
                st.rerun()

def dashboard_page():
    if not st.session_state.get('analysis_results'):
        st.warning("Please run an analysis from the Home page first.", icon="ℹ️")
        if st.button("⬅️ Back to Home"): st.session_state.page = "🏠 Home"; st.rerun()
        return
    
    res = st.session_state.analysis_results
    st.title("📊 SEO Analysis Dashboard")
    with st.container(border=True):
        st.markdown(f"##### Results for: `{res['url']}`")
        score = res.get('seo_score', 0)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Score", f"{score}/100", "🎉" if score >= 80 else "👍" if score >= 60 else "🤔")
        load_time = res['performance']['load_time']
        c2.metric("Load Time", f"{load_time:.2f}s" if isinstance(load_time, float) else "N/A")
        c3.metric("Word Count", f"{res['content']['word_count']:,}")
        c4.metric("Page Size", f"{res['performance']['size_kb']:.1f} KB")
    
    tab_onpage, tab_content, tab_tech, tab_semantic, tab_structure, tab_competitor = st.tabs([
        "📄 On-Page", "📑 Content", "🛠️ Technical", "🧠 Semantic Analysis", "📐 Structure Optimizer", "🔍 Competitor Comparison"
    ])

    with tab_onpage:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Title Tag")
            title = res['on_page']['title']
            display_check("Optimal Length", title['status'])
            st.metric("Length", f"{title['length']} chars", f"Optimal: {CONFIG['seo_scores']['title_range'][0]}-{CONFIG['seo_scores']['title_range'][1]}")
            st.text_area("Title Text", title['text'], height=50, disabled=True)
        with c2:
            st.subheader("Meta Description")
            desc = res['on_page']['description']
            display_check("Optimal Length", desc['status'])
            st.metric("Length", f"{desc['length']} chars", f"Optimal: {CONFIG['seo_scores']['desc_range'][0]}-{CONFIG['seo_scores']['desc_range'][1]}")
            st.text_area("Description Text", desc['text'], height=100, disabled=True)
        
        st.subheader("Headings Structure")
        display_check("Single H1 Tag", len(res['on_page']['headings']['h1']) == 1)
        if res['on_page']['multiple_h1']:
            st.warning("Multiple H1 tags found. It's generally best practice to have only one H1 per page for SEO and accessibility.", icon="⚠️")
        if not res['on_page']['h1_present']:
            st.warning("No H1 tag found. An H1 tag is crucial for signaling your page's main topic to search engines.", icon="⚠️")
        with st.expander("View all headings"): st.json({k: v for k, v in res['on_page']['headings'].items() if v})
        
        st.subheader("Image SEO (Alt Text)")
        st.info(res['on_page']['image_alt_status'])


    with tab_content:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Content Analysis")
            st.metric("Readability (Flesch)", f"{res['content']['readability_score']:.1f}", "Aim > 60")
            st.metric("Word Count", f"{res['content']['word_count']:,}", f"Min: {CONFIG['seo_scores']['min_word_count']}")
        with c2:
            st.subheader("Top 20 Keywords (from your content)")
            if res['content']['keywords']:
                df = pd.DataFrame(res['content']['keywords'], columns=['Keyword', 'Frequency'])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else: st.info("No significant keywords found in content.")

    with tab_tech:
        st.subheader("Technical SEO Checks")
        tech = res['technical']
        c1, c2 = st.columns(2)
        with c1:
            display_check("Uses HTTPS", tech.get('https', False))
            display_check("Canonical Tag Present", tech.get('canonical', False))
            display_check("Mobile Viewport Meta Tag", tech.get('viewport', False))
            display_check("Robots.txt Present", tech.get('robots_txt_present', False))
        with c2:
            display_check("Schema Markup Present (JSON-LD)", tech.get('schema_present', False))
            display_check("Sitemap.xml Present (Root Domain)", tech.get('sitemap_xml_present', False))
            st.metric("Internal Links Found", tech.get('internal_links_count', 0), f"Aim > {CONFIG['seo_scores']['min_internal_links']}")
            display_check("Mobile Friendly Meta (Basic Check)", tech.get('mobile_friendly_meta', False))
            

    with tab_semantic:
        st.header("🧠 Semantic Analysis Results")
        if st.session_state.get('semantic_results'):
            sem_res = st.session_state.semantic_results
            st.subheader("Content Topic Summary")
            st.info(sem_res.get('summary', 'No summary available.'))

            st.subheader("Key Entities Identified")
            if sem_res.get('entities'):
                st.table(pd.DataFrame(sem_res['entities'], columns=["Entity"]))
            else:
                st.info("No key entities identified.")

            st.subheader("Suggested LSI (Latent Semantic Indexing) Keywords")
            if sem_res.get('lsi_keywords'):
                st.table(pd.DataFrame(sem_res['lsi_keywords'], columns=["LSI Keyword"]))
            else:
                st.info("No LSI keywords suggested.")
            
            st.subheader("Readability Score (from Semantic Analysis)")
            st.metric("Flesch Reading Ease", f"{sem_res.get('readability_score', 0):.1f}")

            if sem_res.get('keyword_coverage'):
                st.subheader("Target Keyword Coverage & Relevance")
                st.markdown(sem_res['keyword_coverage'])
            else:
                st.info("No specific target keywords were provided for coverage analysis.")

        else:
            st.info("Semantic analysis not performed. Please run a full analysis from the Home page.", icon="ℹ️")

    with tab_structure:
        st.header("📐 Structure Optimization Suggestions")
        if st.session_state.get('structure_results'):
            st.subheader("Optimized Content Structure (Markdown)")
            st.markdown(st.session_state.structure_results)
            st.info("Copy and paste this markdown into your content editor. Review for accuracy and flow.", icon="📝")
        else:
            st.info("Structure optimization not performed. Please run a full analysis from the Home page.", icon="ℹ️")

    with tab_competitor:
        st.header("🔍 Competitor Comparison")
        if st.session_state.get('competitor_results'):
            comp_results_df = pd.DataFrame(st.session_state.competitor_results)
            
            if not comp_results_df.empty:
                st.subheader("Key SEO Metrics Comparison")
                display_df = comp_results_df.drop(columns=['Main Content Text'], errors='ignore')
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                if "SEO Score" in comp_results_df.columns:
                    st.subheader("SEO Score Comparison")
                    fig = px.bar(comp_results_df, x='URL', y='SEO Score', title='SEO Score vs. Competitors', color='URL')
                    st.plotly_chart(fig, use_container_width=True)
                
                if "Word Count" in comp_results_df.columns:
                    st.subheader("Word Count Comparison")
                    fig = px.bar(comp_results_df, x='URL', y='Word Count', title='Word Count vs. Competitors', color='URL')
                    st.plotly_chart(fig, use_container_width=True)

                if len(comp_results_df) > 1 and 'Main Content Text' in comp_results_df.columns:
                    st.subheader("Content Similarity (Target vs. Competitors)")
                    your_url = comp_results_df.iloc[0]['URL']
                    your_content = comp_results_df.iloc[0]['Main Content Text']
                    similarity_data = []

                    for i in range(1, len(comp_results_df)):
                        comp_url = comp_results_df.iloc[i]['URL']
                        comp_content = comp_results_df.iloc[i]['Main Content Text']
                        if your_content and comp_content:
                            similarity = get_content_similarity(your_content, comp_content)
                            similarity_data.append({"Competitor URL": comp_url, "Similarity Score": similarity})
                        else:
                            similarity_data.append({"Competitor URL": comp_url, "Similarity Score": "N/A (content not available)"})
                    
                    if similarity_data:
                        sim_df = pd.DataFrame(similarity_data)
                        st.dataframe(sim_df, use_container_width=True, hide_index=True)
                        st.info("Higher similarity scores indicate more overlapping topics/keywords.", icon="💡")
                    else:
                        st.info("Could not calculate content similarity (insufficient content or URLs).")
            else:
                st.info("No competitor data available for comparison.")
        else:
            st.info("Competitor analysis not performed. Please run a full analysis from the Home page and provide competitor URLs or target keywords.", icon="ℹ️")
    
    if st.button("⬅️ Back to Home"): st.session_state.page = "🏠 Home"; st.rerun()


def keyword_research_page():
    st.title("🔑 Keyword Research")
    if not st.session_state.api_key_configured:
        st.warning("This tool requires a Gemini API Key.", icon="⚠️")
        return
    with st.form("keyword_form"):
        seed_keyword = st.text_input("Enter a seed keyword", placeholder="e.g., 'digital marketing'")
        submitted = st.form_submit_button("Generate Keywords", type="primary", use_container_width=True)
        if submitted and seed_keyword:
            with st.spinner("AI is performing keyword research..."):
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    prompt = f"""
                    Act as a keyword research expert. For the seed keyword '{seed_keyword}', generate:
                    1. A list of 20+ related keywords.
                    2. Categorize them by search intent (Informational, Commercial, Navigational, Transactional).
                    3. Suggest 5 compelling article titles based on these keywords.
                    Present the output as a JSON object with keys: 'related_keywords', 'intent_categories', 'suggested_titles'.
                    The 'related_keywords' should be a list of strings.
                    The 'intent_categories' should be a dictionary where keys are intents and values are lists of keywords.
                    The 'suggested_titles' should be a list of strings.
                    """
                    response = model.generate_content(prompt)
                    json_response = response.text.strip()
                    if json_response.startswith("```json") and json_response.endswith("```"):
                        json_response = json_response[len("```json"):-len("```")]
                    st.session_state.keyword_results = json.loads(json_response)
                except json.JSONDecodeError as e:
                    st.error(f"AI response was not valid JSON for keyword research: {e}. Raw response: {response.text}")
                except Exception as e:
                    st.error(f"Failed to generate keywords: {e}")
    if st.session_state.keyword_results:
        res = st.session_state.keyword_results
        st.subheader("Suggested Article Titles")
        for title in res.get('suggested_titles', []): st.markdown(f"- {title}")
        st.subheader("Keywords by Search Intent")
        for intent, keywords in res.get('intent_categories', {}).items():
            with st.expander(f"{intent.capitalize()} ({len(keywords)} keywords)"):
                st.table(pd.DataFrame(keywords, columns=["Keyword"]))

def content_rewriter_page():
    st.title("✍️ Content Rewriter")
    if not st.session_state.api_key_configured:
        st.warning("This tool requires a Gemini API Key.", icon="⚠️")
        return
    with st.form("rewriter_form"):
        text_to_rewrite = st.text_area("Paste your text here to rewrite", height=200)
        style = st.selectbox("Choose a rewriting style", ["Professional", "Conversational", "Benefit-Focused", "Concise", "Expanded"])
        submitted = st.form_submit_button("Rewrite Content", type="primary", use_container_width=True)
        if submitted and text_to_rewrite:
            with st.spinner(f"AI is rewriting in a {style} style..."):
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    prompt = f"Rewrite the following text in 3 different variations of a '{style}' tone. Present as a JSON object with keys 'variation_1', 'variation_2', 'variation_3'.\n\nText:\n{text_to_rewrite}"
                    response = model.generate_content(prompt)
                    json_response = response.text.strip()
                    if json_response.startswith("```json") and json_response.endswith("```"):
                        json_response = json_response[len("```json"):-len("```")]
                    st.session_state.rewriter_results = json.loads(json_response)
                except json.JSONDecodeError as e:
                    st.error(f"AI response was not valid JSON for content rewriter: {e}. Raw response: {response.text}")
                except Exception as e:
                    st.error(f"Failed to rewrite content: {e}")
    if st.session_state.rewriter_results:
        st.subheader("Rewritten Variations")
        for i, (key, value) in enumerate(st.session_state.rewriter_results.items()):
            st.markdown(f"--- \n **Variation {i+1}**")
            st.markdown(value)

def backlink_auditor_page():
    st.title("🔗 Backlink Auditor")
    st.info("This tool scrapes Ahrefs' Free Backlink Checker for an overview of the top 100 backlinks. Due to Ahrefs' anti-scraping measures, this may not always work and is for basic insights only.", icon="ℹ️")
    with st.form("backlink_form"):
        domain_to_check = st.text_input("Enter domain to check backlinks for", placeholder="example.com or https://example.com")
        submitted = st.form_submit_button("Audit Backlinks", type="primary", use_container_width=True)
        if submitted and domain_to_check:
            try:
                parsed_uri = urlparse(domain_to_check)
                clean_domain = parsed_uri.netloc or parsed_uri.path
                if clean_domain.startswith('www.'):
                    clean_domain = clean_domain[4:]
                clean_domain = clean_domain.split('/')[0]

                if not clean_domain:
                    st.error("Invalid domain or URL entered. Please enter a valid domain like 'example.com'.")
                    return

                with st.spinner(f"Scraping Ahrefs for {clean_domain}..."):
                    url = f"https://ahrefs.com/backlink-checker?target={quote_plus(clean_domain)}"
                    headers = {'User-Agent': CONFIG['api']['user_agent']} 
                    response = requests.get(url, headers=headers, timeout=CONFIG['api']['timeout'])
                    
                    if response.status_code == 403:
                        st.error(f"Failed to fetch backlinks: Ahrefs blocked the request (403 Forbidden). This can happen with frequent use or if their anti-scraping measures are active.")
                        st.session_state.backlink_results = []
                        return

                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    links = []
                    table = soup.find('table', class_='ahrefs-table') or soup.find('table', class_='_table_1q14v_1')
                    
                    if not table:
                        st.warning(f"No backlink data table found for '{clean_domain}' on Ahrefs. The domain might be new, have no backlinks, or Ahrefs' HTML structure has changed. Consider checking manually on Ahrefs' free tool.")
                        st.session_state.backlink_results = []
                        return
                    
                    for row in table.select("tbody > tr"):
                        cells = row.find_all("td")
                        if len(cells) >= 6:
                            try:
                                referring_page = cells[0].get_text(strip=True)
                                dr = cells[1].get_text(strip=True)
                                ur = cells[2].get_text(strip=True)
                                traffic = cells[3].get_text(strip=True)
                                anchor_backlink = cells[5].get_text(strip=True)
                                
                                links.append({
                                    "Referring Page": referring_page,
                                    "DR": dr,
                                    "UR": ur,
                                    "Traffic": traffic,
                                    "Anchor and Backlink": anchor_backlink,
                                })
                            except IndexError:
                                continue
                    st.session_state.backlink_results = links
                    if not links:
                        st.info(f"Successfully checked '{clean_domain}', but no backlinks were found in the results table from Ahrefs.")

            except requests.exceptions.RequestException as e:
                st.error(f"Failed to fetch backlinks due to a network error: {e}. Ensure the domain is correct and try again.")
            except Exception as e:
                st.error(f"An unexpected error occurred during backlink audit: {e}") 

    if st.session_state.get('backlink_results'):
        links = st.session_state.backlink_results
        if links:
            st.subheader(f"Found {len(links)} Backlinks")
            df = pd.DataFrame(links)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.subheader("Top 15 Anchor Texts")
            df['Anchor Text'] = df['Anchor and Backlink'].apply(lambda x: x.split('›')[0].strip() if '›' in x else x.strip())
            anchor_counts = Counter(word for word in df['Anchor Text'] if word and word.lower() not in ['image', 'link'] and len(word) > 2).most_common(15)
            if anchor_counts:
                anchor_df = pd.DataFrame(anchor_counts, columns=['Anchor Text', 'Count'])
                fig = px.bar(anchor_df, x='Count', y='Anchor Text', orientation='h', title='Anchor Text Distribution')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No meaningful anchor text data was found in the returned backlinks.")
        else:
            st.info("No backlink results to display.")


def ai_copilot_page():
    st.title("🤖 AI SEO Copilot")
    if not st.session_state.api_key_configured:
        st.warning("This tool requires a Gemini API Key.", icon="⚠️")
        return
    if not st.session_state.get('analysis_results'):
        st.info("Run an analysis from the Home page first for contextual advice.", icon="ℹ️")
    
    for msg in st.session_state.copilot_history:
        with st.chat_message(msg["role"]): 
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask for SEO advice..."):
        st.session_state.copilot_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): 
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    context_parts = []
                    if st.session_state.analysis_results:
                        context_parts.append(f"Primary SEO Audit Data (JSON): {json.dumps(st.session_state.analysis_results, indent=2)}")
                    if st.session_state.semantic_results:
                        context_parts.append(f"Semantic Analysis Data (JSON): {json.dumps(st.session_state.semantic_results, indent=2)}")
                    if st.session_state.structure_results:
                        context_parts.append(f"Structure Optimization Suggestion (Markdown): {st.session_state.structure_results}")
                    if st.session_state.competitor_results:
                        comp_summary = []
                        for comp in st.session_state.competitor_results:
                            comp_summary.append({k: v for k, v in comp.items() if k != 'Main Content Text'})
                        context_parts.append(f"Competitor Analysis Summary (JSON): {json.dumps(comp_summary, indent=2)}")
                    if st.session_state.keyword_results:
                        context_parts.append(f"Keyword Research Data (JSON): {json.dumps(st.session_state.keyword_results, indent=2)}")

                    context = "\n\n".join(context_parts) if context_parts else "No specific audit data available. Please run an analysis first."

                    full_prompt = f"You are an expert SEO analyst. Based on the following available data (if any), answer the user's question. Provide actionable and concise SEO advice. If specific data is missing for the user's query, state that you need more information.\n\n---Available Data---\n{context}\n\n---User's Question---\n{prompt}"
                    
                    response = model.generate_content(full_prompt)
                    ai_response = response.text
                    st.markdown(ai_response)
                    st.session_state.copilot_history.append({"role": "assistant", "content": ai_response})
                except Exception as e:
                    st.error(f"Could not generate response from AI: {e}")

# --- Dummy pages for direct access from sidebar if not part of main flow ---
def semantic_analysis_page():
    st.title("🧠 Semantic Analysis (Stand-alone)")
    st.warning("This page is mainly for direct access. Semantic analysis is automatically performed during the full 'Home' page analysis.", icon="ℹ️")
    if st.session_state.get('semantic_results'):
        with st.expander("View Last Semantic Analysis Results", expanded=True):
            sem_res = st.session_state.semantic_results
            st.subheader("Content Topic Summary")
            st.info(sem_res.get('summary', 'No summary available.'))
            st.subheader("Key Entities Identified")
            if sem_res.get('entities'): st.table(pd.DataFrame(sem_res['entities'], columns=["Entity"]))
            else: st.info("No key entities identified.")
            st.subheader("Suggested LSI (Latent Semantic Indexing) Keywords")
            if sem_res.get('lsi_keywords'): st.table(pd.DataFrame(sem_res['lsi_keywords'], columns=["LSI Keyword"]))
            else: st.info("No LSI keywords suggested.")
            st.subheader("Readability Score")
            st.metric("Flesch Reading Ease", f"{sem_res.get('readability_score', 0):.1f}")
            if sem_res.get('keyword_coverage'):
                st.subheader("Target Keyword Coverage & Relevance")
                st.markdown(sem_res['keyword_coverage'])
    else:
        st.info("No semantic analysis data available. Please run an audit from the Home page first.")

def structure_optimizer_page():
    st.title("📐 Structure Optimizer (Stand-alone)")
    st.warning("This page is mainly for direct access. Structure optimization is automatically performed during the full 'Home' page analysis.", icon="ℹ️")
    if st.session_state.get('structure_results'):
        with st.expander("View Last Structure Optimization Suggestions", expanded=True):
            st.subheader("Optimized Content Structure (Markdown)")
            st.markdown(st.session_state.structure_results)
            st.info("Copy and paste this markdown into your content editor. Review for accuracy and flow.", icon="📝")
    else:
        st.info("No structure optimization data available. Please run an audit from the Home page first.")

def competitor_analysis_page():
    st.title("🔍 Competitor Analysis (Stand-alone)")
    st.warning("This page is mainly for direct access. Competitor analysis is automatically performed during the full 'Home' page analysis if URLs are provided.", icon="ℹ️")
    if st.session_state.get('competitor_results'):
        with st.expander("View Last Competitor Analysis Results", expanded=True):
            comp_results_df = pd.DataFrame(st.session_state.competitor_results)
            if not comp_results_df.empty:
                st.subheader("Key SEO Metrics Comparison")
                display_df = comp_results_df.drop(columns=['Main Content Text'], errors='ignore')
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                if "SEO Score" in comp_results_df.columns:
                    st.subheader("SEO Score Comparison")
                    fig = px.bar(comp_results_df, x='URL', y='SEO Score', title='SEO Score vs. Competitors', color='URL')
                    st.plotly_chart(fig, use_container_width=True)
                
                if "Word Count" in comp_results_df.columns:
                    st.subheader("Word Count Comparison")
                    fig = px.bar(comp_results_df, x='URL', y='Word Count', title='Word Count vs. Competitors', color='URL')
                    st.plotly_chart(fig, use_container_width=True)

                if len(comp_results_df) > 1 and 'Main Content Text' in comp_results_df.columns:
                    st.subheader("Content Similarity (Target vs. Competitors)")
                    your_content = comp_results_df.iloc[0]['Main Content Text']
                    similarity_data = []

                    for i in range(1, len(comp_results_df)):
                        comp_url = comp_results_df.iloc[i]['URL']
                        comp_content = comp_results_df.iloc[i]['Main Content Text']
                        if your_content and comp_content:
                            similarity = get_content_similarity(your_content, comp_content)
                            similarity_data.append({"Competitor URL": comp_url, "Similarity Score": similarity})
                        else:
                            similarity_data.append({"Competitor URL": comp_url, "Similarity Score": "N/A (content not available)"})
                    
                    if similarity_data:
                        sim_df = pd.DataFrame(similarity_data)
                        st.dataframe(sim_df, use_container_width=True, hide_index=True)
                        st.info("Higher similarity scores indicate more overlapping topics/keywords.", icon="💡")
                    else:
                        st.info("Could not calculate content similarity (insufficient content or URLs).")
            else:
                st.info("No competitor data available for comparison.")
    else:
        st.info("No competitor analysis data available. Please run an audit from the Home page first with competitor URLs.")


# --- MAIN APPLICATION RUNNER ---

def main():
    load_css()
    if 'nltk_setup_done' not in st.session_state:
        st.session_state.nltk_setup_done = setup_nltk()
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
            "🔗 Backlink Auditor": backlink_auditor_page,
            "🤖 AI SEO Copilot": ai_copilot_page,
            "--- Reports ---": None,
            "🧠 Semantic Analysis": semantic_analysis_page,
            "📐 Structure Optimizer": structure_optimizer_page,
            "🔍 Competitor Analysis": competitor_analysis_page,
        }

        for page_name, page_func in PAGES.items():
            if page_func is None:
                 st.markdown(f"**{page_name}**")
                 continue
            if st.button(page_name, use_container_width=True, key=f"nav_{page_name}"):
                st.session_state.page = page_name
                st.rerun()

        st.markdown("---")
        st.info(f"App Version: {CONFIG['app']['version']}")
        st.markdown("[GitHub Repo](https://github.com/your-repo-link) | [Feedback](mailto:your.email@example.com)", unsafe_allow_html=True)

    page_to_render = PAGES.get(st.session_state.page, home_page)
    if page_to_render:
        page_to_render()

if __name__ == "__main__":
    main()
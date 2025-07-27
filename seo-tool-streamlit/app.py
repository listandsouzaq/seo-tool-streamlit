# --- IMPORTS ---
import streamlit as st
import os
import time
import requests
import pandas as pd
import google.generativeai as genai
import plotly.express as px
from collections import Counter
import json
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, quote_plus
import nltk
from nltk.corpus import stopwords
import textstat
import extruct
from w3lib.html import get_base_url
import warnings
from typing import Dict, Any, Optional, List, Tuple

# --- PAGE CONFIGURATION & SECRETS ---
st.set_page_config(
    page_title="AI SEO Analyst",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings('ignore')

CONFIG = {
    "app": {
        "title": "AI SEO Analyst",
        "icon": "🔮",
        "version": "6.0"
    },
    "theme": {
        "bg_color": "#0f172a",
        "text_color": "#e2e8f0",
        "card_bg": "#1e293b",
        "accent_color": "#38bdf8",
        "primary_color": "#818cf8"
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
        "max_load_time": 2.5
    },
    "api": {
        "user_agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        "timeout": 15
    },
    "backlinks": {
        "url": "https://www.openlinkprofiler.org/r/{domain}"
    }
}

# --- API KEY MANAGEMENT ---
def get_api_key():
    # Load from Streamlit secrets if present, else from environment as fallback.
    api_key = st.secrets.get("GEMINI_API_KEY") if "GEMINI_API_KEY" in st.secrets else os.environ.get("GEMINI_API_KEY")
    return api_key if api_key else None

def show_secrets_note():
    st.info("""
    **ℹ️ To use AI features, add your Google Gemini API key in `.streamlit/secrets.toml` or use the Streamlit Cloud Secrets Manager.**
    Example:
    ```toml
    # .streamlit/secrets.toml
    GEMINI_API_KEY = "your-gemini-api-key"
    ```
    """)

# --- INITIALIZATION ---
@st.cache_resource
def setup_nltk():
    for resource, path in [("punkt", "tokenizers/punkt"), ("stopwords", "corpora/stopwords")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)
setup_nltk()

def init_session_state():
    defaults = {
        'analysis_results': None,
        'page': "🏠 Home",
        'ai_suggestions': {},
        'copilot_history': [],
        'backlink_df': None,
        'api_key_configured': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- CSS ---
def load_css():
    theme = CONFIG['theme']
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, [class*="st-"], .st-emotion-cache-10trblm {{
            font-family: 'Inter', sans-serif;
            color: {theme['text_color']};
        }}
        .stApp {{ background-color: {theme['bg_color']}; }}
        div[data-testid="stSidebar"] > div:first-child {{ background-color: {theme['card_bg']}; }}
        .st-emotion-cache-1y4p8pa {{ background: linear-gradient(135deg, {theme['primary_color']} 0%, {theme['accent_color']} 100%); color: white; border: none; }}
        div[data-testid="stMetric"] {{
            background-color: {theme['card_bg']}; border-radius: 0.75rem; padding: 20px; border-left: 5px solid {theme['primary_color']};
        }}
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            background-color: {theme['card_bg']}; color: {theme['primary_color']};
            border-bottom: 2px solid {theme['primary_color']};
        }}
    </style>
    """, unsafe_allow_html=True)

def create_barchart(data: List[Tuple[str, int]], title: str) -> Optional[px.bar]:
    if not data: return None
    theme = CONFIG['theme']
    labels, values = zip(*data)
    fig = px.bar(x=values, y=labels, orientation='h', title=title, color_discrete_sequence=[theme['primary_color']])
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': theme['text_color'], 'family': 'Inter'},
        title_font_size=18
    )
    return fig

# --- SEOAuditor Class ---
class SEOAuditor:
    def __init__(self, url: Optional[str] = None, html_content: Optional[str] = None):
        self.url = url
        self.html_content = html_content
        self.results: Dict[str, Any] = {}
        self.soup: Optional[BeautifulSoup] = None
        self.headers = {'User-Agent': CONFIG['api']['user_agent']}

    def run_audit(self) -> Optional[Dict[str, Any]]:
        try:
            if self.url:
                if not (self.url.startswith('http://') or self.url.startswith('https://')):
                    raise ValueError("Invalid URL. Please include http:// or https://")
                start_time = time.time()
                response = requests.get(self.url, headers=self.headers, timeout=CONFIG['api']['timeout'])
                response.raise_for_status()
                self.html_content = response.text
                perf = {'load_time': time.time() - start_time, 'size_kb': len(response.content) / 1024, 'status': response.status_code}
            elif self.html_content:
                self.url = "local_content"
                perf = {'load_time': 'N/A', 'size_kb': len(self.html_content.encode('utf-8')) / 1024, 'status': 'N/A'}
            else:
                raise ValueError("Either a URL or HTML content must be provided.")

            self.soup = BeautifulSoup(self.html_content, 'html.parser')
            for element in self.soup(["script", "style", "noscript", "iframe", "header", "footer"]):
                element.decompose()

            self.results = {
                'performance': perf,
                'on_page': self._check_on_page(),
                'content': self._analyze_content(),
                'technical': self._check_technical(),
                'url': self.url
            }
            self.results['seo_score'] = self._calculate_seo_score()
            return self.results
        except Exception as e:
            st.error(f"Analysis Failed: {e}", icon="🚨")
            return None

    def _check_on_page(self) -> Dict[str, Any]:
        title_tag = self.soup.find('title')
        title_text = title_tag.string.strip() if title_tag and title_tag.string else ""
        desc_tag = self.soup.find('meta', attrs={'name': 'description'})
        desc_text = desc_tag.get('content', '').strip() if desc_tag else ""
        images = self.soup.find_all('img')
        total_img = len(images)
        alt_img = sum(1 for img in images if img.get('alt', '').strip())
        return {
            'title': {'text': title_text, 'length': len(title_text), 'status': CONFIG['seo_scores']['title_range'][0] <= len(title_text) <= CONFIG['seo_scores']['title_range'][1]},
            'description': {'text': desc_text, 'length': len(desc_text), 'status': CONFIG['seo_scores']['desc_range'][0] <= len(desc_text) <= CONFIG['seo_scores']['desc_range'][1]},
            'headings': {f'h{i}': [h.get_text(strip=True) for h in self.soup.find_all(f'h{i}')] for i in range(1, 7)},
            'images': {'total': total_img, 'with_alt': alt_img, 'missing_alt': total_img - alt_img}
        }

    def _analyze_content(self) -> Dict[str, Any]:
        main_content_area = self.soup.find('main') or self.soup.find('article') or self.soup.body
        text = main_content_area.get_text(separator=' ', strip=True) if main_content_area else ""
        words = [word for word in nltk.word_tokenize(text.lower()) if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]
        return {
            'full_text': text,
            'word_count': len(words),
            'readability_score': textstat.flesch_reading_ease(text) if text else 0,
            'keywords': Counter(filtered_words).most_common(20)
        }

    def _check_technical(self) -> Dict[str, Any]:
        base_url = get_base_url(self.html_content, self.url) if self.url != "local_content" else None
        schema_data = extruct.extract(self.html_content, base_url=base_url, syntaxes=['json-ld'], errors='ignore')
        internal_links, external_links = set(), set()
        base_netloc = urlparse(self.url).netloc if self.url != "local_content" else ""
        for link in self.soup.find_all('a', href=True):
            href = link['href']
            if not href or href.startswith(('#', 'mailto:', 'tel:')): continue
            abs_url = urljoin(self.url, href)
            parsed_url = urlparse(abs_url)
            if parsed_url.netloc == base_netloc: internal_links.add(abs_url)
            elif parsed_url.netloc: external_links.add(abs_url)
        return {
            'https': self.url.startswith('https://') if self.url != "local_content" else False,
            'canonical': bool(self.soup.find('link', rel='canonical')),
            'schema_present': bool(schema_data.get('json-ld')),
            'viewport': bool(self.soup.find('meta', attrs={'name': 'viewport'})),
            'internal_links': len(internal_links),
            'external_links': len(external_links)
        }

    def _calculate_seo_score(self) -> int:
        score, max_score = 0, 100
        weights = CONFIG['seo_scores']['weights']
        scores = CONFIG['seo_scores']
        res = self.results
        checks = {
            'title_ok': res['on_page']['title']['status'],
            'desc_ok': res['on_page']['description']['status'],
            'h1_ok': len(res['on_page']['headings']['h1']) == 1,
            'https': res['technical']['https'],
            'content_long': res['content']['word_count'] >= scores['min_word_count'],
            'alt_tags_ok': res['on_page']['images']['total'] > 0 and (res['on_page']['images']['with_alt'] / max(1, res['on_page']['images']['total'])) >= 0.9,
            'readability_ok': res['content']['readability_score'] >= scores['min_readability'],
            'canonical': res['technical']['canonical'],
            'schema': res['technical']['schema_present'],
            'viewport': res['technical']['viewport'],
            'internal_links': res['technical']['internal_links'] > 2,
            'load_time_ok': isinstance(res['performance']['load_time'], float) and res['performance']['load_time'] < scores['max_load_time']
        }
        for check, is_ok in checks.items():
            if is_ok:
                score += weights[check]
        return min(score, max_score)

# --- AI Integration ---
def run_ai_analysis(prompt: str) -> Dict[str, Any]:
    if not st.session_state.get('api_key_configured', False):
        return {'status': 'error', 'message': 'AI features are disabled. Configure your API key.'}
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt, request_options={"timeout": 120})
        return {'status': 'success', 'response': response.text}
    except Exception as e:
        return {'status': 'error', 'message': f'AI analysis error: {e}'}

# --- Backlink Scraper ---
def fetch_backlinks(domain: str) -> Optional[pd.DataFrame]:
    try:
        url = CONFIG['backlinks']['url'].format(domain=quote_plus(domain))
        response = requests.get(url, headers={'User-Agent': CONFIG['api']['user_agent']}, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'table-striped'})
        if not table:
            return None
        rows = table.find_all('tr')[1:]
        data = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 5:
                data.append({
                    'Source URL': cols[1].text.strip(),
                    'Anchor Text': cols[2].text.strip(),
                    'Link Power*Trust': cols[4].text.strip(),
                    'Date Found': cols[5].text.strip()
                })
        return pd.DataFrame(data) if data else None
    except Exception as e:
        st.error(f"Failed to scrape backlinks: {e}", icon="🔥")
        return None

# --- APP PAGES ---
def home_page():
    st.image("https://i.imgur.com/gY52B8a.png", use_container_width=True)
    st.title(CONFIG['app']['title'])
    st.markdown("Enter a URL or paste raw HTML for a deep SEO audit.")

    col1, col2 = st.columns(2)
    with col1:
        url_input = st.text_input("Website URL", placeholder="https://www.example.com", key="url_input")
        if st.button("Analyze URL", use_container_width=True, type="primary"):
            if url_input:
                with st.spinner("Performing deep analysis..."):
                    st.session_state.analysis_results = SEOAuditor(url=url_input).run_audit()
                    if st.session_state.analysis_results:
                        st.session_state.page = "📊 Dashboard"
                        st.rerun()
            else:
                st.warning("Please enter a URL.")
    with col2:
        html_input = st.text_area("Or Paste Raw HTML/Text", height=150, key="html_input")
        if st.button("Analyze Content", use_container_width=True):
            if html_input:
                with st.spinner("Analyzing your content..."):
                    st.session_state.analysis_results = SEOAuditor(html_content=html_input).run_audit()
                    if st.session_state.analysis_results:
                        st.session_state.page = "📊 Dashboard"
                        st.rerun()
            else:
                st.warning("Please paste content.")

def dashboard_page():
    if not st.session_state.get('analysis_results'):
        st.warning("Please run an analysis from Home.")
        if st.button("⬅️ Back to Home"):
            st.session_state.page = "🏠 Home"
            st.rerun()
        return

    res = st.session_state.analysis_results
    st.title("📊 SEO Analysis Dashboard")
    st.markdown(f"##### Results for: `{res['url']}`")

    score = res['seo_score']
    score_emoji = "🎉" if score >= 80 else "👍" if score >= 60 else "🤔"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Score", f"{score}/100", f"{score_emoji}")
    load_time = res['performance']['load_time']
    c2.metric("Load Time", f"{load_time:.2f}s" if isinstance(load_time, float) else "N/A", delta_color="inverse")
    c3.metric("Word Count", res['content']['word_count'])
    c4.metric("H1 Tags", len(res['on_page']['headings'].get('h1', [])), delta_color="off")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📄 On-Page SEO", "📑 Content & Keywords", "🛠️ Technical SEO"])
    with tab1:
        st.subheader("On-Page Elements")
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**Title:** {res['on_page']['title']['text']}", icon="🏷️")
            st.metric("Title Length", f"{res['on_page']['title']['length']} chars", "Optimal" if res['on_page']['title']['status'] else "Check Length")
            st.info(f"**Meta Description:** {res['on_page']['description']['text']}", icon="📝")
            st.metric("Description Length", f"{res['on_page']['description']['length']} chars", "Optimal" if res['on_page']['description']['status'] else "Check Length")
        with c2:
            st.metric("Images with Alt Text", f"{res['on_page']['images']['with_alt']} / {res['on_page']['images']['total']}", f"{res['on_page']['images']['missing_alt']} missing")
            heading_counts = {k.upper(): len(v) for k, v in res['on_page']['headings'].items() if v}
            if heading_counts:
                st.plotly_chart(create_barchart(list(heading_counts.items()), "Heading Distribution"), use_container_width=True)
    with tab2:
        st.subheader("Content Analysis")
        c1, c2, c3 = st.columns(3)
        readability = res['content']['readability_score']
        c1.metric("Word Count", res['content']['word_count'])
        c2.metric("Readability (Flesch)", f"{readability:.1f}", "Easy" if readability >= 60 else "Hard")
        c3.metric("Page Size", f"{res['performance']['size_kb']:.1f} KB")
        if res['content']['keywords']:
            st.plotly_chart(create_barchart(res['content']['keywords'][:15], "Top 15 Keywords"), use_container_width=True)
    with tab3:
        st.subheader("Technical Checklist")
        tech = res['technical']
        tech_items = {
            "HTTPS Enabled": tech['https'], "Canonical Tag": tech['canonical'],
            "Schema Markup": tech['schema_present'], "Mobile Viewport": tech['viewport']
        }
        cols = st.columns(len(tech_items))
        for col, (label, status) in zip(cols, tech_items.items()):
            col.metric(label, "✅ Yes" if status else "❌ No", delta_color="off")
        c1, c2 = st.columns(2)
        c1.metric("Internal Links", tech['internal_links'])
        c2.metric("External Links", tech['external_links'])

def backlink_auditor_page():
    st.title("🔗 Backlink Auditor")
    st.warning("Disclaimer: This tool scrapes a third-party website. It may be slow or fail if the source website changes.", icon="⚠️")
    domain_input = st.text_input("Enter Domain to Audit (e.g., example.com)", key="backlink_domain")
    if st.button("🔗 Fetch Backlinks", type="primary", use_container_width=True):
        if domain_input:
            domain_to_check = urlparse(domain_input).netloc or domain_input.split('/')[0]
            with st.spinner(f"Scraping backlinks for {domain_to_check}..."):
                df = fetch_backlinks(domain_to_check)
                st.session_state.backlink_df = df
                if df is not None:
                    st.success(f"Success! Found {len(df)} backlinks for {domain_to_check}.")
                else:
                    st.error("Could not find backlink data for the domain.")
        else:
            st.warning("Please enter a domain.")
    if 'backlink_df' in st.session_state and st.session_state.backlink_df is not None:
        df = st.session_state.backlink_df
        st.markdown("### Discovered Backlinks")
        st.dataframe(df, use_container_width=True)
        st.markdown("### Top 15 Anchor Texts")
        anchor_counts = Counter(df['Anchor Text'].dropna().replace('', 'No Anchor Text'))
        if anchor_counts:
            chart_data = anchor_counts.most_common(15)
            st.plotly_chart(create_barchart(chart_data, "Anchor Text Distribution"), use_container_width=True)

def keyword_research_page():
    st.title("🔑 AI Keyword Research Tool")
    if not st.session_state.api_key_configured:
        show_secrets_note()
        return
    keyword = st.text_input("Seed Keyword", key="keyword_input", placeholder="e.g., 'sustainable energy'")
    if st.button("💡 Get Keyword Ideas", type="primary", use_container_width=True):
        if not keyword.strip():
            st.error("Please enter a keyword.")
            return
        with st.spinner(f"AI brainstorming for '{keyword}'..."):
            prompt = f"""
            You are a world-class SEO strategist. For the seed keyword '{keyword}', generate 20 related keyword ideas.
            Categorize them by user search intent (Informational, Commercial, Navigational, Transactional).
            Provide a compelling, SEO-friendly title suggestion for an article targeting each keyword.
            Return the result ONLY as a valid JSON array of objects. Each object must have three keys: "keyword_idea", "search_intent", and "suggested_title".
            Example: [{{"keyword_idea": "what is content marketing", "search_intent": "Informational", "suggested_title": "What is Content Marketing? A Beginner's Guide"}}]
            """
            st.session_state.ai_suggestions['keywords'] = run_ai_analysis(prompt)
    if 'keywords' in st.session_state.ai_suggestions:
        result = st.session_state.ai_suggestions['keywords']
        if result['status'] == 'success':
            try:
                json_str = result['response'].strip().lstrip("```json").rstrip("```")
                keyword_data = json.loads(json_str)
                df = pd.DataFrame(keyword_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            except (json.JSONDecodeError, KeyError) as e:
                st.error(f"AI returned unexpected format: {e}", icon="😞")
                st.code(result['response'], language='text')
        else:
            st.error(result['message'])

def content_rewriter_page():
    st.title("✍️ AI Content Rewriter")
    if not st.session_state.api_key_configured:
        show_secrets_note()
        return
    original_content = st.session_state.analysis_results['content']['full_text'] if st.session_state.get('analysis_results') else ""
    if not original_content:
        st.info("Run an analysis on the Home page to load content here, or paste your own.")
    content_to_rewrite = st.text_area("Original Content", value=original_content, height=250, key="rewrite_area")
    if st.button("✨ Rewrite Content", type="primary", use_container_width=True):
        if not content_to_rewrite.strip():
            st.error("Please enter some content.")
            return
        with st.spinner("AI crafting new versions of your content..."):
            prompt = f"""
            As an expert SEO copywriter, rewrite the following text in three distinct styles: Professional, Conversational, and Benefit-Focused.
            Use markdown headings for each style.
            ORIGINAL TEXT:
            ---
            {content_to_rewrite}
            """
            st.session_state.ai_suggestions['rewriter'] = run_ai_analysis(prompt)
    if 'rewriter' in st.session_state.ai_suggestions:
        result = st.session_state.ai_suggestions['rewriter']
        if result['status'] == 'success':
            st.markdown("### AI-Generated Content")
            st.markdown(result['response'], unsafe_allow_html=True)
        else:
            st.error(result['message'])

def seo_copilot_page():
    st.title("🤖 SEO Copilot")
    if not st.session_state.api_key_configured:
        show_secrets_note()
        return
    if not st.session_state.get('analysis_results'):
        st.warning("Please run an analysis first.", icon="ℹ️")
        return
    res = st.session_state.analysis_results
    audit_summary = json.dumps({k: res[k] for k in ['url', 'seo_score', 'on_page', 'content', 'technical', 'performance'] if k in res}, default=str)
    for author, text in st.session_state.copilot_history:
        with st.chat_message(author):
            st.markdown(text)
    if user_question := st.chat_input("Ask about your audit, e.g., 'Give me 3 actionable tips.'"):
        st.session_state.copilot_history.append(("user", user_question))
        st.chat_message("user").write(user_question)
        with st.spinner("Copilot is thinking..."):
            prompt = f"""
            You are an SEO Copilot: helpful, concise, and expert.
            Based ONLY on the provided JSON audit data, answer the user's question.
            Do not make up information. If the data isn't available, state that clearly.

            AUDIT DATA:
            {audit_summary}

            USER QUESTION:
            {user_question}
            """
            response = run_ai_analysis(prompt)
            answer = response['response'] if response['status'] == 'success' else response['message']
            st.session_state.copilot_history.append(("assistant", answer))
            st.chat_message("assistant").write(answer)
            st.rerun()

# --- MAIN APP ---
def main():
    init_session_state()
    api_key = get_api_key()
    if api_key and not st.session_state.api_key_configured:
        try:
            genai.configure(api_key=api_key)
            genai.get_model('gemini-1.5-flash')
            st.session_state.api_key_configured = True
        except Exception:
            st.session_state.api_key_configured = False
    elif not api_key:
        st.session_state.api_key_configured = False

    load_css()

    with st.sidebar:
        st.image("https://i.imgur.com/gY52B8a.png", use_container_width=True)
        st.header(CONFIG["app"]["title"])
        PAGES = {
            "🏠 Home": home_page,
            "📊 Dashboard": dashboard_page,
            "🔗 Backlink Auditor": backlink_auditor_page,
            "🔑 Keyword Research": keyword_research_page,
            "✍️ Content Rewriter": content_rewriter_page,
            "🤖 SEO Copilot": seo_copilot_page
        }
        st.session_state.page = st.radio(
            "Navigation", list(PAGES.keys()), key="nav_radio", label_visibility="collapsed"
        )
        st.markdown("---")
        if st.session_state.api_key_configured:
            st.success("AI Features Enabled", icon="✅")
        else:
            st.warning("AI Features Disabled", icon="⚠️")
            show_secrets_note()
        st.markdown(f"""
        <div style="font-size: 0.8rem; color: #94a3b8; position: absolute; bottom: 10px;">
             <strong>v{CONFIG['app']['version']}</strong>
        </div>
        """, unsafe_allow_html=True)
    PAGES[st.session_state.page]()

if __name__ == "__main__":
    main()
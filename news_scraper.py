#!/usr/bin/env python3
"""
news_scraper_ml.py
Enhanced with proper ML models for ministry classification and sentiment analysis
"""

import os
from dotenv import load_dotenv
from supabase import create_client

# Load .env file (local development)
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    print("⚠️ WARNING: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set in environment.")
    print("Supabase upserts will fail until keys are provided.")
    supabase = None
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def save_to_supabase(article_dict):
    """Inserts/upserts one scraped article into Supabase database."""
    if supabase is None:
        print("⚠️ Supabase client not initialized. Skipping upsert.")
        return None

    try:
        payload = {
            "id": article_dict.get("id"),
            "source_type": article_dict.get("source_type"),
            "source_name": article_dict.get("source_name"),
            "timestamp": article_dict.get("timestamp"),
            "language": article_dict.get("language"),
            "title": article_dict.get("title"),
            "content": article_dict.get("content"),
            "summary": article_dict.get("summary"),
            "url": article_dict.get("url"),
            "author": article_dict.get("author"),
            "ministries": article_dict.get("ministries") or [],
            "ministry_scores": article_dict.get("ministry_scores") or {},
            "sentiment_score": article_dict.get("sentiment_score"),
            "sentiment_label": article_dict.get("sentiment_label"),
            "keywords": article_dict.get("keywords") or [],
            "metadata": article_dict.get("metadata") or {}
        }

        res = supabase.table("news_articles").upsert(payload).execute()
        print("✅ Saved to Supabase:", payload.get("title", "")[:120])
        return res
    except Exception as e:
        print("❌ Supabase insert error:", e)
        return None


# ------------------- Standard imports -------------------
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
import re
from flask import Flask, jsonify, request
from flask_cors import CORS
import time
import logging
from urllib.parse import urljoin, urlparse
import hashlib
import random
from collections import defaultdict
from pathlib import Path
import threading

# ML imports
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
import torch
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Create data directory for storing news
DATA_DIR = Path("news_data")
DATA_DIR.mkdir(exist_ok=True)

# ------------------- ML Models Initialization -------------------

class MLModels:
    """Initialize and manage ML models for classification"""
    
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"🤖 Initializing ML models on {'GPU' if self.device == 0 else 'CPU'}...")
        
        # 1. Sentiment Analysis - Using multilingual model
        try:
            print("Loading sentiment analysis model...")
            model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            if self.device == 0:
                self.sentiment_model = self.sentiment_model.cuda()
            
            self.sentiment_model.eval()
            print("✅ Sentiment model loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading sentiment model: {e}")
            self.sentiment_model = None
            self.sentiment_tokenizer = None
        
        # 2. Zero-shot classification for ministries
        try:
            print("Loading zero-shot classification model...")
            self.ministry_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=self.device
            )
            print("✅ Ministry classifier loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading ministry classifier: {e}")
            self.ministry_classifier = None
        
        # Ministry labels for classification
        self.ministry_labels = [
            "health and medical services",
            "finance and economy", 
            "education and schools",
            "sports and games",
            "international affairs and diplomacy",
            "agriculture and farming",
            "politics and government",
            "defence and military",
            "general news"
        ]
        
        self.ministry_mapping = {
            "health and medical services": "health",
            "finance and economy": "finance",
            "education and schools": "education",
            "sports and games": "sports",
            "international affairs and diplomacy": "international_affairs",
            "agriculture and farming": "agriculture",
            "politics and government": "politics",
            "defence and military": "defence",
            "general news": "general"
        }
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using transformer model"""
        if not text or len(text) < 10:
            return 0.0, "neutral"
        
        if self.sentiment_model is None or self.sentiment_tokenizer is None:
            # Fallback to neutral
            return 0.0, "neutral"
        
        try:
            # Truncate text to avoid token limit
            text_sample = text[:512]
            
            # Tokenize
            inputs = self.sentiment_tokenizer(
                text_sample, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            if self.device == 0:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Model outputs: negative, neutral, positive
            scores_cpu = scores.cpu().numpy()[0]
            
            negative_score = scores_cpu[0]
            neutral_score = scores_cpu[1]
            positive_score = scores_cpu[2]
            
            # Calculate sentiment score (-1 to 1)
            sentiment_score = positive_score - negative_score
            
            # Determine label
            max_score = max(negative_score, neutral_score, positive_score)
            if max_score == positive_score:
                sentiment_label = "positive"
            elif max_score == negative_score:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            return round(float(sentiment_score), 3), sentiment_label
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return 0.0, "neutral"
    
    def classify_ministries(self, title, content, top_k=3):
        """Classify article into ministries using zero-shot classification"""
        
        if self.ministry_classifier is None:
            return ["general"], {"general": 1.0}
        
        try:
            # Combine title (more important) and beginning of content
            text = f"{title}. {title}. {content[:300]}"
            
            # Use zero-shot classification
            result = self.ministry_classifier(
                text,
                candidate_labels=self.ministry_labels,
                multi_label=True
            )
            
            # Extract results
            ministry_scores = {}
            top_ministries = []
            
            for label, score in zip(result['labels'], result['scores']):
                # Only keep scores above threshold
                if score > 0.15:  # Lower threshold to get more classifications
                    ministry_key = self.ministry_mapping.get(label, "general")
                    ministry_scores[ministry_key] = round(float(score), 3)
                    
                    if len(top_ministries) < top_k:
                        top_ministries.append(ministry_key)
            
            # If no ministry found, default to general
            if not top_ministries:
                return ["general"], {"general": 1.0}
            
            return top_ministries, ministry_scores
            
        except Exception as e:
            logger.error(f"Ministry classification error: {e}")
            return ["general"], {"general": 1.0}

# Initialize ML models globally
print("🔄 Loading ML models (this may take a minute)...")
ml_models = MLModels()
print("✅ ML models ready!\n")

# ------------------- Configuration -------------------

NEWS_SOURCES = {
    "hindi": [
        {"name": "Aaj Tak", "url": "https://www.aajtak.in"},
        {"name": "ABP News", "url": "https://www.abplive.com"},
        {"name": "News18 India", "url": "https://hindi.news18.com"},
        {"name": "Zee News", "url": "https://zeenews.india.com/hindi"},
        {"name": "NDTV India", "url": "https://ndtv.in/livetv-ndtvindia"}
    ],
    "kannada": [
        {"name": "TV9 Kannada", "url": "https://tv9kannada.com"},
        {"name": "Public TV", "url": "https://publictv.in"},
        {"name": "NewsFirst Kannada", "url": "https://newsfirstlive.com"},
        {"name": "Kannada Prabha", "url": "https://www.kannadaprabha.com"},
        {"name": "Udayavani", "url": "https://www.udayavani.com"}
    ],
    "bengali": [
        {"name": "News18 Bangla", "url": "https://bengali.news18.com"},
        {"name": "TV9 Bangla", "url": "https://www.tv9bangla.com"},
        {"name": "Zee 24 Ghanta", "url": "https://zeenews.india.com/bengali"},
        {"name": "Anandabazar", "url": "https://www.anandabazar.com"},
        {"name": "Ei Samay", "url": "https://eisamay.indiatimes.com"}
    ]
}

LANGUAGE_MAPPING = {
    "hindi": "Hindi",
    "kannada": "Kannada",
    "bengali": "Bengali"
}

SUPPORTED_MINISTRIES = [
    "health", "finance", "education", "sports",
    "international_affairs", "agriculture", "politics", "defence", "general"
]

# ------------------- Storage and Article model -------------------

class NewsStorage:
    """Handle local JSON storage for news articles"""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def get_daily_filename(self, date=None):
        """Get filename for daily news storage"""
        if date is None:
            date = datetime.now()
        return self.data_dir / f"news_{date.strftime('%Y-%m-%d')}.json"

    def load_daily_news(self, date=None):
        """Load news for a specific date"""
        filename = self.get_daily_filename(date)
        if filename.exists():
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading daily news: {str(e)}")
        return {"articles": [], "metadata": {"date": datetime.now().strftime('%Y-%m-%d'), "total": 0}}

    def save_daily_news(self, articles, date=None):
        """Save news articles for a specific date"""
        filename = self.get_daily_filename(date)

        # Load existing data
        existing_data = self.load_daily_news(date)

        # Convert articles and avoid duplicates
        existing_hashes = {article.get('metadata', {}).get('content_hash') for article in existing_data['articles']}
        new_articles = []

        for article in articles:
            article_dict = article.to_schema_dict() if hasattr(article, 'to_schema_dict') else article
            content_hash = article_dict.get('metadata', {}).get('content_hash')

            if content_hash not in existing_hashes:
                new_articles.append(article_dict)
                existing_hashes.add(content_hash)

        # Combine with existing articles
        all_articles = existing_data['articles'] + new_articles

        # Sort by timestamp
        all_articles.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        # Prepare data to save
        data_to_save = {
            "metadata": {
                "date": (date or datetime.now()).strftime('%Y-%m-%d'),
                "total": len(all_articles),
                "last_updated": datetime.now().isoformat(),
                "new_articles_added": len(new_articles),
                "ministries": list(set(article.get('ministries', [None])[0] for article in all_articles if article.get('ministries'))),
                "languages": list(set(article.get('language') for article in all_articles if article.get('language'))),
                "sources": list(set(article.get('source_name') for article in all_articles if article.get('source_name')))
            },
            "articles": all_articles
        }

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(new_articles)} new articles to {filename}")
            return len(new_articles)
        except Exception as e:
            logger.error(f"Error saving daily news: {str(e)}")
            return 0


class NewsArticle:
    def __init__(self, title, content, url, source_name, language_key, author=None, website_section=None):
        self.title = title.strip()
        self.content = content.strip()
        self.url = url
        self.source_name = source_name
        self.language_key = language_key
        self.language = LANGUAGE_MAPPING.get(language_key, language_key.title())
        self.author = author
        self.website_section = website_section
        self.timestamp = datetime.now().isoformat() + "Z"
        self.scrape_time = self.timestamp

        # Generate content hash
        self.content_hash = hashlib.md5((title.lower() + content.lower()).encode('utf-8')).hexdigest()

        # Generate unique ID
        self.id = hashlib.md5((self.title + self.url + self.content_hash).encode('utf-8')).hexdigest()

        # Use ML models for analysis
        print(f"🔍 Analyzing: {title[:50]}...")
        self.ministries, self.ministry_scores = ml_models.classify_ministries(title, content)
        self.sentiment_score, self.sentiment_label = ml_models.analyze_sentiment(f"{title}. {content}")
        print(f"   └─ Ministries: {self.ministries}, Sentiment: {self.sentiment_label}")
        
        self.keywords = self.extract_keywords(title, content)
        self.summary = self.generate_summary(title, content)

    def extract_keywords(self, title, content):
        """Extract keywords from title and content"""
        text = f"{title} {content}".lower()

        # Enhanced stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'been', 'have', 'has', 'had',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'do', 'does', 'did', 'get', 'got',
            'का', 'की', 'के', 'में', 'को', 'से', 'और', 'या', 'है', 'हैं', 'था', 'थे', 'यह', 'वह',
            'এর', 'এই', 'ও', 'ছিল', 'আছে', 'করে', 'হয়', 'থেকে', 'দিয়ে', 'তার', 'যে', 'কি',
            'ಇದು', 'ಆದ', 'ಮತ್ತು', 'ಅಥವಾ', 'ಇದೆ', 'ಆಗಿದೆ', 'ಮಾಡಿ', 'ಆ', 'ಈ', 'ಒಂದು'
        }

        # Extract words
        words = re.findall(r'\b\w{3,}\b', text)
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        # Get top keywords by frequency
        word_freq = defaultdict(float)
        title_words = set(re.findall(r'\b\w{3,}\b', title.lower()))

        for word in keywords:
            weight = 3.0 if word in title_words else 1.0
            word_freq[word] += weight

        # Filter common uninformative words
        common_uninformative = {'news', 'said', 'today', 'new', 'also', 'more', 'time', 'year', 'years'}
        word_freq = {word: freq for word, freq in word_freq.items() if word not in common_uninformative}

        # Return top 10 keywords
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_keywords[:10]]

    def generate_summary(self, title, content, max_length=200):
        """Generate a summary of the article"""
        if not content or len(content) < 100:
            return title[:max_length]

        # Simple extractive summarization
        sentences = re.split(r'[।।\.\!\?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return title[:max_length]

        summary = title + ". "
        for sentence in sentences[:2]:
            if len(summary + sentence) < max_length:
                summary += sentence + ". "
            else:
                break

        return summary.strip()

    def to_schema_dict(self):
        """Convert to JSON schema format"""
        return {
            "id": self.id,
            "source_type": "web_scraper",
            "source_name": self.source_name,
            "timestamp": self.timestamp,
            "language": self.language,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "url": self.url,
            "author": self.author,
            "ministries": self.ministries,
            "ministry_scores": self.ministry_scores,
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
            "keywords": self.keywords,
            "metadata": {
                "scrape_time": self.scrape_time,
                "website_section": self.website_section,
                "content_hash": self.content_hash
            }
        }


class NewsScraper:
    def __init__(self):
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        ]

        self.headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.storage = NewsStorage(DATA_DIR)

    def extract_articles(self, soup, base_url, source_name, language_key):
        """Extract articles from webpage"""
        articles = []
        selectors = [
            'article', '[class*="story"]', '[class*="article"]', '[class*="news"]',
            '[class*="post"]', 'h1, h2, h3, h4', 'a[title]', '[class*="headline"]'
        ]

        found_titles = set()

        for selector in selectors:
            try:
                elements = soup.select(selector)
                for element in elements[:20]:
                    try:
                        title = self.extract_title(element)
                        if not title or len(title) < 15 or len(title) > 200:
                            continue

                        title_hash = hashlib.md5(title.lower().encode('utf-8')).hexdigest()
                        if title_hash in found_titles:
                            continue
                        found_titles.add(title_hash)

                        content = self.extract_content(element)
                        if not content:
                            content = f"News article: {title}"

                        article_url = self.extract_url(element, base_url)
                        author = self.extract_author(element)
                        section = self.extract_section(element)

                        article = NewsArticle(
                            title=title,
                            content=content,
                            url=article_url,
                            source_name=source_name,
                            language_key=language_key,
                            author=author,
                            website_section=section
                        )

                        articles.append(article)

                        if len(articles) >= 15:
                            break

                    except Exception as e:
                        logger.debug(f"Error processing element: {str(e)}")
                        continue

                if len(articles) >= 15:
                    break

            except Exception as e:
                logger.debug(f"Error with selector {selector}: {str(e)}")
                continue

        return articles

    def extract_title(self, element):
        """Extract title from element"""
        title = ""
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            title = element.get_text(strip=True)
        elif element.name == 'a' and element.get('title'):
            title = element.get('title').strip()
        elif element.name == 'article':
            title_elem = element.find(['h1', 'h2', 'h3', 'h4', 'h5'])
            if title_elem:
                title = title_elem.get_text(strip=True)
        else:
            title_elem = element.find(['h1', 'h2', 'h3', 'h4', 'h5']) or element
            title = title_elem.get_text(strip=True)

        title = re.sub(r'\s+', ' ', title).strip()
        title = re.sub(r'^[^a-zA-Z\u0900-\u097F\u0980-\u09FF\u0C80-\u0CFF]*', '', title)

        skip_patterns = ['advertisement', 'sponsored', 'live:', 'watch:', 'video:', 'photo:']
        if any(skip in title.lower() for skip in skip_patterns):
            return None

        return title

    def extract_content(self, element):
        """Extract content from element"""
        content = ""

        parent = element.parent
        if parent:
            desc_elem = parent.find(['p', 'div', 'span'],
                                   class_=re.compile(r'(summary|excerpt|desc|intro|lead)', re.I))
            if desc_elem:
                desc_text = desc_elem.get_text(strip=True)
                if len(desc_text) > 50:
                    content = desc_text

        if not content and element.name in ['h1', 'h2', 'h3', 'h4', 'h5']:
            paragraphs = []
            for next_elem in element.find_next_siblings(['p', 'div'], limit=3):
                next_text = next_elem.get_text(strip=True)
                if len(next_text) > 30:
                    paragraphs.append(next_text)
            if paragraphs:
                content = ' '.join(paragraphs)

        if content:
            content = re.sub(r'\s+', ' ', content).strip()
            if len(content) > 800:
                content = content[:800] + "..."

        return content

    def extract_url(self, element, base_url):
        """Extract URL from element"""
        link_elem = None
        if element.name == 'a':
            link_elem = element
        else:
            link_elem = element.find('a', href=True)

        if link_elem and link_elem.get('href'):
            href = link_elem['href']
            if href.startswith('http'):
                return href
            elif href.startswith('/'):
                return urljoin(base_url, href)

        return base_url

    def extract_author(self, element):
        """Extract author from element"""
        author_selectors = ['[class*="author"]', '[class*="byline"]', '[class*="writer"]']

        for selector in author_selectors:
            author_elem = element.find(selector) or (element.parent and element.parent.find(selector))
            if author_elem:
                author_text = author_elem.get_text(strip=True)
                if len(author_text) < 100 and len(author_text) > 2:
                    return author_text

        return None

    def extract_section(self, element):
        """Extract website section from element"""
        section_selectors = ['[class*="section"]', '[class*="category"]', '[class*="topic"]']

        for selector in section_selectors:
            section_elem = element.find(selector) or (element.parent and element.parent.find(selector))
            if section_elem:
                section_text = section_elem.get_text(strip=True)
                if len(section_text) < 50 and len(section_text) > 2:
                    return section_text

        return None

    def scrape_source(self, source_info, language_key):
        """Scrape a single news source"""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                logger.info(f"Scraping {source_info['name']} ({language_key}): {source_info['url']}")
                time.sleep(random.uniform(1, 3))

                response = self.session.get(source_info['url'], timeout=30)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                for unwanted in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    unwanted.decompose()

                articles = self.extract_articles(soup, source_info['url'], source_info['name'], language_key)

                if articles:
                    saved_count = self.storage.save_daily_news(articles)

                    for article in articles:
                        try:
                            article_dict = article.to_schema_dict()
                            save_to_supabase(article_dict)
                        except Exception as e:
                            logger.warning(f"Failed to save article to Supabase: {e}")

                    logger.info(f"Successfully scraped {len(articles)} articles from {source_info['name']}")
                    return articles
                else:
                    logger.warning(f"No articles found for {source_info['name']}")

            except Exception as e:
                logger.warning(f"Error scraping {source_info['name']} (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(5)

        return []

    def scrape_all_sources(self, language=None):
        """Scrape all sources or specific language"""
        all_articles = []
        languages_to_scrape = [language] if language else list(NEWS_SOURCES.keys())

        for lang in languages_to_scrape:
            if lang not in NEWS_SOURCES:
                logger.warning(f"Language '{lang}' not supported")
                continue

            for source in NEWS_SOURCES[lang]:
                try:
                    articles = self.scrape_source(source, lang)
                    all_articles.extend(articles)
                    time.sleep(2)
                except Exception as e:
                    logger.error(f"Error scraping {source['name']}: {str(e)}")
                    continue

        return all_articles


# Initialize scraper
scraper = NewsScraper()

# ------------------- Flask API Endpoints -------------------

@app.route('/api/news', methods=['GET'])
def get_news():
    """Get news articles with filtering"""
    try:
        language = request.args.get('language', None)
        ministry = request.args.get('ministry', None)
        date_str = request.args.get('date', None)
        fresh = request.args.get('fresh', 'true').lower() == 'true'

        date = None
        if date_str:
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

        # Get articles from Supabase
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo
        three_hours_ago = datetime.now(ZoneInfo("Asia/Kolkata")) - timedelta(hours=3)

        query = (
            supabase.table("news_articles")
            .select("*")
            .gte("timestamp", three_hours_ago.isoformat())
        )

        data = query.execute()
        articles_data = data.data or []

        # Filter by ministry if specified
        if ministry and ministry != 'all':
            articles_data = [
                article for article in articles_data
                if ministry in article.get('ministries', [])
            ]

        # Filter by language if specified
        if language:
            articles_data = [
                article for article in articles_data
                if article.get('language', '').lower() == LANGUAGE_MAPPING.get(language, language).lower()
            ]

        # Sort by timestamp
        articles_data.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        return jsonify({
            "articles": articles_data,
            "total": len(articles_data),
            "filters": {
                "language": language,
                "ministry": ministry,
                "date": date_str,
                "fresh": fresh
            },
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in news endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route('/api/scrape', methods=['POST'])
def scrape_news():
    """Manually trigger news scraping"""
    try:
        language = request.json.get('language') if request.json else None
        articles = scraper.scrape_all_sources(language)

        return jsonify({
            "message": "Scraping completed",
            "articles_scraped": len(articles),
            "language": language,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in scrape endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route('/api/languages', methods=['GET'])
def get_languages():
    """Get supported languages"""
    return jsonify({
        "languages": list(NEWS_SOURCES.keys()),
        "language_mapping": LANGUAGE_MAPPING,
        "total": len(NEWS_SOURCES)
    })


@app.route('/api/ministries', methods=['GET'])
def get_ministries():
    """Get supported ministries"""
    ministry_info = {}
    for ministry in SUPPORTED_MINISTRIES:
        ministry_info[ministry] = {
            "name": ministry.replace('_', ' ').title()
        }

    return jsonify({
        "ministries": ministry_info,
        "total": len(SUPPORTED_MINISTRIES)
    })


@app.route('/api/sources', methods=['GET'])
def get_sources():
    """Get news sources"""
    return jsonify(NEWS_SOURCES)


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about stored articles"""
    try:
        days = int(request.args.get('days', 7))

        stats = {
            "total_articles": 0,
            "by_language": defaultdict(int),
            "by_ministry": defaultdict(int),
            "by_source": defaultdict(int),
            "by_sentiment": defaultdict(int),
            "daily_counts": []
        }

        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            daily_data = scraper.storage.load_daily_news(date)

            daily_count = len(daily_data['articles'])
            stats["total_articles"] += daily_count
            stats["daily_counts"].append({
                "date": date.strftime('%Y-%m-%d'),
                "count": daily_count
            })

            for article in daily_data['articles']:
                language = article.get('language', 'Unknown')
                stats["by_language"][language] += 1

                ministries = article.get('ministries', ['general'])
                for ministry in ministries:
                    stats["by_ministry"][ministry] += 1

                source = article.get('source_name', 'Unknown')
                stats["by_source"][source] += 1

                sentiment = article.get('sentiment_label', 'neutral')
                stats["by_sentiment"][sentiment] += 1

        stats["by_language"] = dict(stats["by_language"])
        stats["by_ministry"] = dict(stats["by_ministry"])
        stats["by_source"] = dict(stats["by_source"])
        stats["by_sentiment"] = dict(stats["by_sentiment"])

        return jsonify({
            "stats": stats,
            "period_days": days,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in stats endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route('/api/search', methods=['GET'])
def search_articles():
    """Search articles by keyword"""
    try:
        query = request.args.get('q', '').lower()
        language = request.args.get('language', None)
        ministry = request.args.get('ministry', None)
        sentiment = request.args.get('sentiment', None)
        days = int(request.args.get('days', 7))

        if not query:
            return jsonify({"error": "Query parameter 'q' is required"}), 400

        matching_articles = []

        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            daily_data = scraper.storage.load_daily_news(date)

            for article in daily_data['articles']:
                title = article.get('title', '').lower()
                content = article.get('content', '').lower()
                keywords = [k.lower() for k in article.get('keywords', [])]

                if (query in title or query in content or
                        any(query in keyword for keyword in keywords)):

                    if language and article.get('language', '').lower() != LANGUAGE_MAPPING.get(language, language).lower():
                        continue

                    if ministry and ministry not in article.get('ministries', []):
                        continue

                    if sentiment and article.get('sentiment_label') != sentiment:
                        continue

                    matching_articles.append(article)

        def relevance_score(article):
            score = 0
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            score += title.count(query) * 3
            score += content.count(query) * 1
            return score

        matching_articles.sort(key=relevance_score, reverse=True)

        return jsonify({
            "articles": matching_articles[:50],
            "total": len(matching_articles),
            "query": query,
            "filters": {
                "language": language,
                "ministry": ministry,
                "sentiment": sentiment,
                "days": days
            },
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0 - Transformer-based ML Multi-language Government News Scraper",
        "features": {
            "supported_languages": list(NEWS_SOURCES.keys()),
            "supported_ministries": SUPPORTED_MINISTRIES,
            "ml_sentiment_analysis": ml_models.sentiment_model is not None,
            "ml_ministry_classification": ml_models.ministry_classifier is not None,
            "local_json_storage": True,
            "supabase_sync": supabase is not None,
            "keyword_extraction": True,
            "auto_summarization": True
        },
        "ml_models": {
            "sentiment_model": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
            "ministry_classifier": "facebook/bart-large-mnli",
            "device": "GPU" if ml_models.device == 0 else "CPU"
        },
        "storage": {
            "directory": str(scraper.storage.data_dir),
            "files_count": len(list(scraper.storage.data_dir.glob("news_*.json")))
        }
    })


@app.route('/api/export', methods=['GET'])
def export_data():
    """Export news data for a specific date range"""
    try:
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        format_type = request.args.get('format', 'json')

        if not start_date_str or not end_date_str:
            return jsonify({"error": "start_date and end_date parameters are required"}), 400

        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

        if start_date > end_date:
            return jsonify({"error": "start_date must be before end_date"}), 400

        all_articles = []
        current_date = start_date

        while current_date <= end_date:
            daily_data = scraper.storage.load_daily_news(current_date)
            all_articles.extend(daily_data['articles'])
            current_date += timedelta(days=1)

        if format_type == 'json':
            return jsonify({
                "articles": all_articles,
                "total": len(all_articles),
                "date_range": {
                    "start": start_date_str,
                    "end": end_date_str
                },
                "export_timestamp": datetime.now().isoformat()
            })

        return jsonify({"error": "Only JSON format is currently supported"}), 400

    except Exception as e:
        logger.error(f"Error in export endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


def auto_scrape():
    """Auto scrape every 1 minute"""
    while True:
        print("\n⏳ Auto scraping triggered (every 1 min)...")
        scraper.scrape_all_sources()
        print("✅ Auto scrape cycle complete. Waiting 1 min...\n")
        time.sleep(60)


# Start auto scraping in background thread
threading.Thread(target=auto_scrape, daemon=True).start()

# Team API endpoints
TEAM_API_KEY = os.environ.get("TEAM_API_KEY")

@app.route("/api/youtube", methods=["POST"])
def add_youtube():
    if request.headers.get("X-API-KEY") != TEAM_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    try:
        supabase.table("youtube_videos").insert(data).execute()
        return jsonify({"message": "YouTube data inserted"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/enews", methods=["POST"])
def add_enews():
    if request.headers.get("X-API-KEY") != TEAM_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    try:
        supabase.table("enews_image_analysis").insert(data).execute()
        return jsonify({"message": "E-News data inserted"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------- Run server -------------------

if __name__ == '__main__':
    print("🚀 Starting Transformer-based ML Multi-language Government News Scraper v4.0...")
    print("\n📰 Supported Languages:")
    for lang_key, lang_name in LANGUAGE_MAPPING.items():
        sources = NEWS_SOURCES.get(lang_key, [])
        print(f"  • {lang_name}: {len(sources)} sources")
    print()

    print("🏛️ Supported Ministries:")
    for ministry in SUPPORTED_MINISTRIES:
        print(f"  • {ministry.replace('_', ' ').title()}")
    print()

    print("🤖 ML Models:")
    print(f"  ✅ Sentiment: {'Loaded' if ml_models.sentiment_model else 'Failed'} (cardiffnlp/twitter-xlm-roberta-base-sentiment)")
    print(f"  ✅ Ministry Classification: {'Loaded' if ml_models.ministry_classifier else 'Failed'} (facebook/bart-large-mnli)")
    print(f"  ✅ Device: {'GPU' if ml_models.device == 0 else 'CPU'}")
    print()

    print("📊 Enhanced Features:")
    print("  ✅ Transformer-based sentiment analysis (multilingual)")
    print("  ✅ Zero-shot ministry classification with confidence scores")
    print("  ✅ Multi-language news scraping")
    print("  ✅ Automatic keyword extraction")
    print("  ✅ Article summarization")
    print("  ✅ Local JSON storage + Supabase sync")
    print("  ✅ RESTful API with filtering and search")
    print("  ✅ Auto-scraping every 1 minute")
    print()

    print("🌐 API Endpoints:")
    print("  • GET  /api/news - Get news articles")
    print("  • POST /api/scrape - Manually trigger scraping")
    print("  • GET  /api/search - Search articles")
    print("  • GET  /api/languages - Get supported languages")
    print("  • GET  /api/ministries - Get supported ministries")
    print("  • GET  /api/sources - Get news sources")
    print("  • GET  /api/stats - Get statistics")
    print("  • GET  /api/export - Export data")
    print("  • GET  /api/health - Health check")
    print("  • POST /api/youtube - Add YouTube data (team)")
    print("  • POST /api/enews - Add E-news data (team)")
    print()

    print("🔧 Server starting on http://localhost:5000")
    app.run(debug=True, port=5000)
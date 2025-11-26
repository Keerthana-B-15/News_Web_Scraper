#!/usr/bin/env python3
"""
news_scraper.py
Full corrected file — scrapes news, saves local JSON, upserts into Supabase,
and exposes Flask API endpoints for fetching, searching and health checks.
"""

import os

# ------------------- Twilio SMS Alert System -------------------
from twilio.rest import Client

TWILIO_SID = os.getenv("TWILIO_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM = os.getenv("TWILIO_FROM", "")  # Twilio phone number
ALERT_TARGET_PHONE = os.getenv("ALERT_TARGET_PHONE", "")  # Your phone number

def send_sms_alert(title, summary, url):
    """Send SMS alert for NEGATIVE sentiment news."""
    if not (TWILIO_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM and ALERT_TARGET_PHONE):
        logger.error("⚠ Twilio credentials missing — SMS not sent.")
        return

    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        msg = client.messages.create(
            body=f"[ALERT] NEGATIVE NEWS\nTitle: {title}\n{summary}\n{url}"[:500],
            from_=TWILIO_FROM,
            to=ALERT_TARGET_PHONE
        )
        logger.info(f"📩 SMS alert sent for: {title}")
    except Exception as e:
        logger.error(f"❌ Failed to send SMS alert: {e}")


from dotenv import load_dotenv
from supabase import create_client
from transformers import pipeline  # ✅ NEW: HuggingFace sentiment model (PyTorch)

# Load .env file (local development)
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    # We won't crash here, but warn — backend will still run (useful for local dev without supabase)
    print("⚠️ WARNING: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set in environment.")
    print("Supabase upserts will fail until keys are provided.")
    supabase = None
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def save_to_supabase(article_dict):
    """
    Inserts/upserts one scraped article into Supabase database.
    article_dict must match the table columns.
    """
    if supabase is None:
        print("⚠️ Supabase client not initialized. Skipping upsert.")
        return None

    try:
        # Ensure types: lists and dicts are properly passed
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
        # Optionally, you can inspect res to ensure success
        if hasattr(res, "status_code"):
            # some versions return different response shapes; be permissive
            pass
        print("✅ Saved to Supabase:", payload.get("title", "")[:120])
        return res
    except Exception as e:
        print("❌ Supabase insert error:", e)
        return None


def save_youtube_to_supabase(video_dict):
    """
    Inserts/upserts scraped YouTube data into Supabase youtube_videos table
    """
    if supabase is None:
        print("⚠️ Supabase not initialized, skipping YouTube insert")
        return None
    try:
        res = supabase.table("youtube_videos").upsert(video_dict).execute()
        print("✅ YouTube video saved:", video_dict.get("title"))
        return res
    except Exception as e:
        print("❌ YouTube insert error:", e)


def save_enews_image_to_supabase(image_dict):
    """
    Inserts image analysis result into enews_image_analysis table
    """
    if supabase is None:
        print("⚠️ Supabase not initialized, skipping E-news image insert")
        return None
    try:
        res = supabase.table("enews_image_analysis").insert(image_dict).execute()
        print("✅ E-news image analysis saved:", image_dict.get("image_name"))
        return res
    except Exception as e:
        print("❌ E-news image insert error:", e)


# ------------------- standard imports for the rest of the scraper -------------------
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Create data directory for storing news
DATA_DIR = Path("news_data")
DATA_DIR.mkdir(exist_ok=True)

# ------------------- Configuration: sources, languages, keywords -------------------

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
    "international_affairs", "agriculture", "politics", "defence"
]


def load_keywords():
    """Load keywords from keywords.json file"""
    try:
        keywords_file = Path("keywords.json")
        if keywords_file.exists():
            with open(keywords_file, 'r', encoding='utf-8') as f:
                keywords_data = json.load(f)
                logger.info(f"Loaded keywords for ministries: {list(keywords_data.keys())}")
                return keywords_data
        else:
            logger.error("keywords.json not found! Please create the file with ministry keywords.")
            return {}
    except Exception as e:
        logger.error(f"Error loading keywords.json: {str(e)}")
        return {}


MINISTRY_KEYWORDS = load_keywords()

PRIORITY_WEIGHTS = {
    "high_priority": 5,
    "medium_priority": 3,
    "low_priority": 1
}

# ------------------- Lightweight Multilingual Sentiment Model (Render Safe) -------------------
print("🌐 Loading ultra-light multilingual sentiment model...")

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

SENTIMENT_MODEL = None
SENTIMENT_TOKENIZER = None
SENTIMENT_PIPELINE = None

try:
    # 🟢 Ultra-light multilingual sentiment model (best for Render Free 512MB)
    model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    print(f"🔄 Trying lightweight model: {model_name}")

    SENTIMENT_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    SENTIMENT_MODEL = AutoModelForSequenceClassification.from_pretrained(model_name)

    SENTIMENT_PIPELINE = pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        tokenizer=SENTIMENT_TOKENIZER,
        device=-1  # force CPU
    )

    print("✅ Loaded ultra-light multilingual sentiment model")
except Exception as e:
    print("❌ Could not load lightweight multilingual model!", e)
    SENTIMENT_PIPELINE = None



# ------------------- Storage and Article model -------------------


class NewsStorage:
    """Handle local JSON storage for news articles according to specified schema"""

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
        """Save news articles for a specific date according to schema"""
        filename = self.get_daily_filename(date)

        # Load existing data
        existing_data = self.load_daily_news(date)

        # Convert articles to schema format and avoid duplicates
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

        # Sort by timestamp (newest first)
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

        # Generate content hash first
        self.content_hash = hashlib.md5((title.lower() + content.lower()).encode('utf-8')).hexdigest()

        # Generate unique ID using title + url + content to avoid duplicates
        self.id = hashlib.md5((self.title + self.url + self.content_hash).encode('utf-8')).hexdigest()

        # Analyze ministries and sentiment
        self.ministries, self.ministry_scores = self.improved_analyze_ministries(title, content)
        self.sentiment_score, self.sentiment_label = self.improved_analyze_sentiment(title, content)
        self.keywords = self.extract_keywords(title, content)
        self.summary = self.generate_summary(title, content)
        # 🔔 Send SMS alert if sentiment is NEGATIVE
        if self.sentiment_label.strip().lower() == "negative":
            send_sms_alert(self.title, self.summary, self.url)


    def improved_analyze_ministries(self, title, content):
        """Improved ministry analysis with better keyword matching and scoring"""
        if not MINISTRY_KEYWORDS:
            return ["general"], {"general": 1.0}

        # Prepare text for analysis - title gets triple weight
        full_text = f"{title} {title} {title} {content}".lower()
        title_text = title.lower()
        content_text = content.lower()

        ministry_scores = {}

        for ministry, priority_groups in MINISTRY_KEYWORDS.items():
            if ministry not in SUPPORTED_MINISTRIES:
                continue

            total_score = 0.0
            title_matches = 0
            content_matches = 0

            # Process each priority level
            for priority_level, keywords in priority_groups.items():
                weight = PRIORITY_WEIGHTS.get(priority_level, 1)

                for keyword in keywords:
                    keyword_lower = keyword.lower().strip()
                    if len(keyword_lower) < 2:  # Skip very short keywords
                        continue

                    # Count occurrences in title and content separately
                    title_count = title_text.count(keyword_lower)
                    content_count = content_text.count(keyword_lower)

                    if title_count > 0:
                        # Title matches get much higher weight
                        title_score = title_count * weight * 10
                        total_score += title_score
                        title_matches += title_count

                    if content_count > 0:
                        # Content matches get standard weight
                        content_score = content_count * weight * 2
                        total_score += content_score
                        content_matches += content_count

            # Calculate normalized score with enhanced formula
            if total_score > 0:
                # Base score calculation
                text_length = max(len(full_text.split()), 50)  # Minimum 50 words for normalization
                base_score = total_score / text_length

                # Boost score based on title matches (title matches are very important)
                title_boost = min(title_matches * 0.5, 0.8)  # Max 80% boost from title
                content_boost = min(content_matches * 0.1, 0.2)  # Max 20% boost from content

                # Final normalized score
                normalized_score = min((base_score + title_boost + content_boost) * 10, 1.0)

                if normalized_score > 0.05:  # Lower threshold for inclusion
                    ministry_scores[ministry] = round(normalized_score, 3)

        # Determine relevant ministries with adjusted threshold
        relevant_ministries = [
            ministry for ministry, score in ministry_scores.items()
            if score > 0.05  # Lowered from 0.1 to 0.05
        ]

        if not relevant_ministries:
            return ["general"], {"general": 1.0}

        # Sort by score and take top 3
        sorted_ministries = sorted(relevant_ministries, key=lambda x: ministry_scores[x], reverse=True)[:3]
        return sorted_ministries, ministry_scores

    def improved_analyze_sentiment(self, title, content):
        """
        Transformer-based sentiment using cardiffnlp/twitter-roberta-base-sentiment-latest.
        No rule-based logic, no TensorFlow.
        """
        if SENTIMENT_MODEL is None:
            return 0.0, "Neutral"

        # Combine title + content, give natural context
        text = f"{title}. {content}".strip()
        if not text:
            return 0.0, "Neutral"

        # Avoid overly long input (model is trained on tweets; keep it tight)
        text = text[:512]

        try:
            pred = SENTIMENT_MODEL(text)[0]
            label = pred["label"].lower()
            score = float(pred["score"])

            # Use mapping style you requested ("Positive"/"Negative"/"Neutral")
            sentiment_label = "Positive" if "positive" in label else (
                "Negative" if "negative" in label else "Neutral"
            )

            # For score, keep sign for negative, positive for positive
            if sentiment_label == "Negative":
                sentiment_score = -round(score, 3)
            elif sentiment_label == "Positive":
                sentiment_score = round(score, 3)
            else:
                sentiment_score = 0.0

            return sentiment_score, sentiment_label

        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return 0.0, "Neutral"

    def extract_keywords(self, title, content):
        """Extract keywords from title and content with improved filtering"""
        text = f"{title} {content}".lower()

        # Enhanced stop words for all languages
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'been', 'have', 'has', 'had',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'do', 'does', 'did', 'get', 'got',
            # Hindi stop words
            'का', 'की', 'के', 'में', 'को', 'से', 'और', 'या', 'है', 'हैं', 'था', 'थे', 'यह', 'वह',
            'इस', 'उस', 'एक', 'दो', 'तीन', 'कर', 'किया', 'करने', 'होगा', 'हो', 'गया', 'गई',
            # Bengali stop words
            'এর', 'এই', 'ও', 'ছিল', 'আছে', 'করে', 'হয়', 'থেকে', 'দিয়ে', 'তার', 'যে', 'কি',
            'একটি', 'দুটি', 'তিনটি', 'করা', 'করেছে', 'হবে', 'হলো', 'গেছে', 'এসেছে',
            # Kannada stop words
            'ಇದು', 'ಆದ', 'ಮತ್ತು', 'ಅಥವಾ', 'ಇದೆ', 'ಆಗಿದೆ', 'ಮಾಡಿ', 'ಆ', 'ಈ', 'ಒಂದು',
            'ಎರಡು', 'ಮೂರು', 'ಮಾಡುವ', 'ಮಾಡಿದ', 'ಆಗುವ', 'ಆಯಿತು', 'ಬಂದಿದೆ'
        }

        # Extract words that are 3+ characters and not stop words
        words = re.findall(r'\b\w{3,}\b', text)
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        # Get top keywords by frequency, prioritizing title words
        word_freq = defaultdict(float)
        title_words = set(re.findall(r'\b\w{3,}\b', title.lower()))

        for word in keywords:
            # Title words get higher weight
            weight = 3.0 if word in title_words else 1.0
            word_freq[word] += weight

        # Filter out very common but uninformative words
        common_uninformative = {'news', 'said', 'today', 'new', 'also', 'more', 'time', 'year', 'years'}
        word_freq = {word: freq for word, freq in word_freq.items() if word not in common_uninformative}

        # Return top 10 keywords
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_keywords[:10]]

    def generate_summary(self, title, content, max_length=200):
        """Generate a summary of the article"""
        if not content or len(content) < 100:
            return title[:max_length]

        # Simple extractive summarization - take first few sentences
        sentences = re.split(r'[।।\.\!\?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return title[:max_length]

        summary = title + ". "
        for sentence in sentences[:2]:  # Take first 2 sentences
            if len(summary + sentence) < max_length:
                summary += sentence + ". "
            else:
                break

        return summary.strip()

    def to_schema_dict(self):
        """Convert to the specified JSON schema format"""
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
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]

        self.headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
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
            'article',
            '[class*="story"]',
            '[class*="article"]',
            '[class*="news"]',
            '[class*="post"]',
            'h1, h2, h3, h4',
            'a[title]',
            '[class*="headline"]',
            '[class*="title"]'
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

        # Clean title
        title = re.sub(r'\s+', ' ', title).strip()
        title = re.sub(r'^[^a-zA-Z\u0900-\u097F\u0980-\u09FF\u0C80-\u0CFF]*', '', title)

        # Skip unwanted titles
        skip_patterns = ['advertisement', 'sponsored', 'live:', 'watch:', 'video:', 'photo:', 'gallery:']
        if any(skip in title.lower() for skip in skip_patterns):
            return None

        return title

    def extract_content(self, element):
        """Extract content from element"""
        content = ""

        # Look for content in parent elements
        parent = element.parent
        if parent:
            desc_elem = parent.find(
                ['p', 'div', 'span'],
                class_=re.compile(r'(summary|excerpt|desc|intro|lead|content|text)', re.I)
            )
            if desc_elem:
                desc_text = desc_elem.get_text(strip=True)
                if len(desc_text) > 50:
                    content = desc_text

        # Look for following paragraphs
        if not content and element.name in ['h1', 'h2', 'h3', 'h4', 'h5']:
            paragraphs = []
            for next_elem in element.find_next_siblings(['p', 'div'], limit=3):
                next_text = next_elem.get_text(strip=True)
                if len(next_text) > 30:
                    paragraphs.append(next_text)
            if paragraphs:
                content = ' '.join(paragraphs)

        # Clean content
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
        author_selectors = [
            '[class*="author"]',
            '[class*="byline"]',
            '[class*="writer"]'
        ]

        for selector in author_selectors:
            author_elem = element.find(selector) or (element.parent and element.parent.find(selector))
            if author_elem:
                author_text = author_elem.get_text(strip=True)
                if 2 < len(author_text) < 100:
                    return author_text

        return None

    def extract_section(self, element):
        """Extract website section from element"""
        section_selectors = [
            '[class*="section"]',
            '[class*="category"]',
            '[class*="topic"]'
        ]

        for selector in section_selectors:
            section_elem = element.find(selector) or (element.parent and element.parent.find(selector))
            if section_elem:
                section_text = section_elem.get_text(strip=True)
                if 2 < len(section_text) < 50:
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

                # Remove unwanted elements
                for unwanted in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    unwanted.decompose()

                articles = self.extract_articles(soup, source_info['url'], source_info['name'], language_key)

                if articles:
                    saved_count = self.storage.save_daily_news(articles)

                    # ---------- Save each article to Supabase ----------
                    for article in articles:
                        try:
                            article_dict = article.to_schema_dict()
                            save_to_supabase(article_dict)
                        except Exception as e:
                            logger.warning(f"Failed to save article to Supabase: {e}")

                    logger.info(
                        f"Successfully scraped {len(articles)} articles from {source_info['name']}, "
                        f"saved {saved_count} new ones"
                    )
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
                    time.sleep(2)  # Be respectful to servers
                except Exception as e:
                    logger.error(f"Error scraping {source['name']}: {str(e)}")
                    continue

        return all_articles


# Initialize scraper
scraper = NewsScraper()

# ------------------- Flask API Endpoints -------------------


@app.route('/api/news', methods=['GET'])
def get_news():
    """Get news articles with filtering options"""
    try:
        language = request.args.get('language', None)
        ministry = request.args.get('ministry', None)
        date_str = request.args.get('date', None)
        fresh = request.args.get('fresh', 'true').lower() == 'true'

        # Parse date
        date = None
        if date_str:
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

        # Always read fresh from Supabase for last 3h
        from datetime import datetime as dt2, timedelta as td2
        from zoneinfo import ZoneInfo
        three_hours_ago = dt2.now(ZoneInfo("Asia/Kolkata")) - td2(hours=3)

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
        keyword_count = 0
        if ministry in MINISTRY_KEYWORDS:
            keyword_count = sum(len(keywords) for keywords in MINISTRY_KEYWORDS[ministry].values())

        ministry_info[ministry] = {
            "name": ministry.replace('_', ' ').title(),
            "keywords_count": keyword_count
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

        # Collect stats from last N days
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
                # Language stats
                language = article.get('language', 'Unknown')
                stats["by_language"][language] += 1

                # Ministry stats
                ministries = article.get('ministries', ['general'])
                for ministry in ministries:
                    stats["by_ministry"][ministry] += 1

                # Source stats
                source = article.get('source_name', 'Unknown')
                stats["by_source"][source] += 1

                # Sentiment stats
                sentiment = article.get('sentiment_label', 'Neutral')
                stats["by_sentiment"][sentiment] += 1

        # Convert defaultdicts to regular dicts
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
    """Search articles by keyword, ministry, or other criteria"""
    try:
        query = request.args.get('q', '').lower()
        language = request.args.get('language', None)
        ministry = request.args.get('ministry', None)
        sentiment = request.args.get('sentiment', None)
        days = int(request.args.get('days', 7))

        if not query:
            return jsonify({"error": "Query parameter 'q' is required"}), 400

        matching_articles = []

        # Search through last N days
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            daily_data = scraper.storage.load_daily_news(date)

            for article in daily_data['articles']:

                # Check if query matches title, content, or keywords
                title = article.get('title', '').lower()
                content = article.get('content', '').lower()
                keywords = [k.lower() for k in article.get('keywords', [])]

                if (query in title or query in content or
                        any(query in keyword for keyword in keywords)):

                    # Apply filters
                    if language and article.get('language', '').lower() != LANGUAGE_MAPPING.get(language, language).lower():
                        continue

                    if ministry and ministry not in article.get('ministries', []):
                        continue

                    if sentiment and article.get('sentiment_label') != sentiment:
                        continue

                    matching_articles.append(article)

        # Sort by relevance (simple scoring based on query occurrence)
        def relevance_score(article):
            score = 0
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()

            # Title matches are more important
            score += title.count(query) * 3
            score += content.count(query) * 1

            return score

        matching_articles.sort(key=relevance_score, reverse=True)

        return jsonify({
            "articles": matching_articles[:50],  # Limit results
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


@app.route('/api/reload-keywords', methods=['POST'])
def reload_keywords():
    """Reload keywords from keywords.json"""
    global MINISTRY_KEYWORDS
    try:
        new_keywords = load_keywords()
        if new_keywords:
            MINISTRY_KEYWORDS = new_keywords
            return jsonify({
                "message": "Keywords reloaded successfully",
                "ministries": list(MINISTRY_KEYWORDS.keys()),
                "total_ministries": len(MINISTRY_KEYWORDS),
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "error": "Failed to load keywords",
                "message": "Please check keywords.json file"
            }), 400
    except Exception as e:
        logger.error(f"Error reloading keywords: {str(e)}")
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
        "version": "2.2 - Multi-language Government News Scraper with Transformer Sentiment",
        "features": {
            "supported_languages": list(NEWS_SOURCES.keys()),
            "supported_ministries": SUPPORTED_MINISTRIES,
            "keywords_loaded": bool(MINISTRY_KEYWORDS),
            "local_json_storage": True,
            "improved_sentiment_analysis": True,
            "transformer_sentiment_model": True,
            "keyword_extraction": True,
            "auto_summarization": True
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
        format_type = request.args.get('format', 'json')  # json or csv

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

        # For other formats, you could implement CSV export here
        return jsonify({"error": "Only JSON format is currently supported"}), 400

    except Exception as e:
        logger.error(f"Error in export endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


import threading


def auto_scrape():
    while True:
        print("\n⏳ Auto scraping triggered (every 1 min)...")
        scraper.scrape_all_sources()
        print("✅ Auto scrape cycle complete. Waiting 1 min...\n")
        time.sleep(60)  # 1 min


# Start auto scraping in background thread
threading.Thread(target=auto_scrape, daemon=True).start()

import os
from flask import request, jsonify  # (already imported above, harmless re-import)

TEAM_API_KEY = os.environ.get("TEAM_API_KEY")


# ---------------- YouTube Data API ----------------
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


# ------------- E-Newspaper Image API --------------
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
    print("🚀 Starting Improved Multi-language Government News Scraper v2.2...")
    print("📰 Supported Languages:")
    for lang_key, lang_name in LANGUAGE_MAPPING.items():
        sources = NEWS_SOURCES.get(lang_key, [])
        print(f"  • {lang_name}: {len(sources)} sources")
    print()

    print("🏛️ Supported Ministries:")
    for ministry in SUPPORTED_MINISTRIES:
        print(f"  • {ministry.replace('_', ' ').title()}")
    print()

    print("📊 Enhanced Features:")
    print("  ✅ Multi-language news scraping (Hindi, Kannada, Bengali)")
    print("  ✅ IMPROVED ministry categorization with better keyword matching")
    print("  ✅ Transformer-based sentiment analysis (cardiffnlp/twitter-roberta-base-sentiment-latest)")
    print("  ✅ Title-weighted keyword analysis")
    print("  ✅ Enhanced stop word filtering")
    print("  ✅ Automatic keyword extraction")
    print("  ✅ Article summarization")
    print("  ✅ Local JSON storage with schema compliance")
    print("  ✅ Deduplication using content hashes")
    print("  ✅ RESTful API with filtering and search")
    print()

    if not Path("keywords.json").exists():
        print("⚠️  WARNING: keywords.json not found!")
        print("   Please make sure your keywords.json file is in the same directory")
        print()
    else:
        print("✅ Keywords loaded from keywords.json")
        if MINISTRY_KEYWORDS:
            total_keywords = sum(
                len(keywords)
                for ministry_data in MINISTRY_KEYWORDS.values()
                for keywords in ministry_data.values()
            )
            print(f"   Loaded {len(MINISTRY_KEYWORDS)} ministries with {total_keywords} total keywords")
        print()

    print("🌐 API Endpoints:")
    print("  • GET  /api/news - Get news articles (supports filtering)")
    print("  • POST /api/scrape - Manually trigger scraping")
    print("  • GET  /api/search - Search articles by keyword")
    print("  • GET  /api/languages - Get supported languages")
    print("  • GET  /api/ministries - Get supported ministries")
    print("  • GET  /api/sources - Get news sources")
    print("  • GET  /api/stats - Get article statistics")
    print("  • GET  /api/export - Export data for date range")
    print("  • POST /api/reload-keywords - Reload keywords file")
    print("  • GET  /api/health - Health check")
    print()

    print("🔧 Server starting on https://news-web-scraper-1.onrender.com/api/news")
    app.run(debug=True, port=5000)

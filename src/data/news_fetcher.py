"""
News Fetcher — pulls headlines from multiple free sources.

Sources:
  1. yfinance Ticker.news (per-symbol news from Yahoo Finance)
  2. Google News RSS (broad market + sector searches)
  3. Yahoo Finance RSS (market-wide)

All sources are free, no API keys required.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_yfinance_news(symbols: List[str], max_per_symbol: int = 5) -> List[dict]:
    """
    Fetch recent news for each symbol via yfinance.
    Returns list of dicts with: symbol, title, link, publisher, published, source.
    """
    import yfinance as yf

    all_news = []

    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            news_items = ticker.news or []

            for item in news_items[:max_per_symbol]:
                # yfinance news items have varying structure across versions
                title = item.get("title", "")
                link = item.get("link", "")
                publisher = item.get("publisher", "")

                # Handle different timestamp formats
                pub_ts = item.get("providerPublishTime", 0)
                if isinstance(pub_ts, (int, float)) and pub_ts > 0:
                    published = datetime.fromtimestamp(pub_ts)
                else:
                    published = datetime.now()

                if title:
                    all_news.append({
                        "symbol": sym,
                        "title": title,
                        "link": link,
                        "publisher": publisher,
                        "published": published,
                        "source": "yfinance",
                    })
        except Exception as e:
            logger.debug(f"News fetch failed for {sym}: {e}")
            continue

    logger.info(f"Fetched {len(all_news)} news items for {len(symbols)} symbols via yfinance")
    return all_news


def fetch_rss_news(
    queries: List[str] = None,
    max_per_query: int = 10,
) -> List[dict]:
    """
    Fetch news from Google News RSS for broad market queries.
    Queries can be sector names, market terms, or specific topics.
    """
    import feedparser

    if queries is None:
        queries = [
            "stock market today",
            "earnings report",
            "FDA approval",
            "merger acquisition",
            "interest rate federal reserve",
            "tech stocks",
            "AI artificial intelligence stocks",
            "cryptocurrency bitcoin",
            "oil price energy",
            "semiconductor chip stocks",
        ]

    all_news = []

    for query in queries:
        try:
            url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)

            for entry in feed.entries[:max_per_query]:
                title = entry.get("title", "")
                link = entry.get("link", "")
                publisher = entry.get("source", {}).get("title", "") if hasattr(entry.get("source", {}), "get") else ""

                # Parse published date
                pub_str = entry.get("published", "")
                try:
                    published = datetime(*entry.published_parsed[:6]) if entry.get("published_parsed") else datetime.now()
                except Exception:
                    published = datetime.now()

                if title:
                    all_news.append({
                        "symbol": "",  # Will be matched later
                        "title": title,
                        "link": link,
                        "publisher": publisher,
                        "published": published,
                        "source": "google_rss",
                        "query": query,
                    })

            # Be polite to Google
            time.sleep(0.5)

        except Exception as e:
            logger.debug(f"RSS fetch failed for '{query}': {e}")
            continue

    logger.info(f"Fetched {len(all_news)} news items from RSS feeds")
    return all_news


def fetch_yahoo_rss(max_items: int = 30) -> List[dict]:
    """Fetch Yahoo Finance top stories via RSS."""
    import feedparser

    urls = [
        "https://finance.yahoo.com/news/rssindex",
        "https://finance.yahoo.com/rss/topstories",
    ]

    all_news = []
    for url in urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_items]:
                title = entry.get("title", "")
                link = entry.get("link", "")

                try:
                    published = datetime(*entry.published_parsed[:6]) if entry.get("published_parsed") else datetime.now()
                except Exception:
                    published = datetime.now()

                if title:
                    all_news.append({
                        "symbol": "",
                        "title": title,
                        "link": link,
                        "publisher": "Yahoo Finance",
                        "published": published,
                        "source": "yahoo_rss",
                    })
        except Exception as e:
            logger.debug(f"Yahoo RSS failed: {e}")

    logger.info(f"Fetched {len(all_news)} items from Yahoo RSS")
    return all_news


def match_news_to_symbols(
    news_items: List[dict],
    known_symbols: List[str],
    company_names: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """
    Match untagged news items to stock symbols by scanning titles
    for ticker mentions and company name keywords.

    Args:
        news_items: List of news dicts (may or may not have 'symbol' set)
        known_symbols: All symbols in universe
        company_names: Optional mapping of symbol -> company name for fuzzy matching
    """
    if company_names is None:
        company_names = _get_default_company_names()

    # Build reverse lookup: keyword -> symbol
    keyword_to_symbol = {}
    for sym in known_symbols:
        keyword_to_symbol[sym] = sym
    for sym, name in company_names.items():
        # Use significant words from company name (skip generic words)
        skip_words = {"inc", "corp", "co", "ltd", "the", "group", "holdings", "and", "of"}
        for word in name.split():
            word_clean = word.strip(",.()").lower()
            if len(word_clean) > 2 and word_clean not in skip_words:
                keyword_to_symbol[word_clean] = sym

    matched = []
    for item in news_items:
        if item.get("symbol"):
            matched.append(item)
            continue

        title_upper = item["title"].upper()
        title_lower = item["title"].lower()
        found_symbols = set()

        # Check for ticker symbols (exact match with word boundary)
        for sym in known_symbols:
            if len(sym) <= 1:
                continue
            # Look for ticker as standalone word
            if f" {sym} " in f" {title_upper} " or f"({sym})" in title_upper:
                found_symbols.add(sym)

        # Check company name keywords
        for keyword, sym in keyword_to_symbol.items():
            if keyword.lower() in title_lower and sym not in found_symbols:
                found_symbols.add(sym)

        if found_symbols:
            for sym in found_symbols:
                new_item = dict(item)
                new_item["symbol"] = sym
                matched.append(new_item)
        else:
            # Keep unmatched items too — they might have sector-level relevance
            matched.append(item)

    return matched


def fetch_all_news(
    symbols: List[str],
    include_rss: bool = True,
    include_yahoo: bool = True,
    max_age_hours: int = 72,
) -> List[dict]:
    """
    Master news fetcher — pulls from all sources and deduplicates.

    Args:
        symbols: Symbols to fetch specific news for
        include_rss: Whether to include Google News RSS
        include_yahoo: Whether to include Yahoo Finance RSS
        max_age_hours: Filter out news older than this

    Returns:
        List of news dicts, sorted by published date (newest first)
    """
    all_news = []

    # 1. Symbol-specific news from yfinance
    all_news.extend(fetch_yfinance_news(symbols))

    # 2. Broad market news from RSS
    if include_rss:
        rss_news = fetch_rss_news()
        rss_news = match_news_to_symbols(rss_news, symbols)
        all_news.extend(rss_news)

    # 3. Yahoo Finance headlines
    if include_yahoo:
        yahoo_news = fetch_yahoo_rss()
        yahoo_news = match_news_to_symbols(yahoo_news, symbols)
        all_news.extend(yahoo_news)

    # Deduplicate by title similarity
    seen_titles = set()
    unique_news = []
    for item in all_news:
        # Normalize title for dedup
        title_key = item["title"].lower().strip()[:80]
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_news.append(item)

    # Filter by age
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    unique_news = [n for n in unique_news if n["published"] >= cutoff]

    # Sort newest first
    unique_news.sort(key=lambda x: x["published"], reverse=True)

    logger.info(
        f"Total unique news items: {len(unique_news)} "
        f"(from {len(all_news)} raw items)"
    )
    return unique_news


def _get_default_company_names() -> Dict[str, str]:
    """Default mapping of ticker -> company name for title matching."""
    return {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Google Alphabet",
        "AMZN": "Amazon",
        "NVDA": "Nvidia",
        "META": "Meta Facebook",
        "TSLA": "Tesla",
        "JPM": "JPMorgan",
        "V": "Visa",
        "JNJ": "Johnson",
        "UNH": "UnitedHealth",
        "HD": "Home Depot",
        "PG": "Procter Gamble",
        "MA": "Mastercard",
        "BAC": "Bank America",
        "XOM": "Exxon",
        "ABBV": "AbbVie",
        "PFE": "Pfizer",
        "COST": "Costco",
        "TMO": "Thermo Fisher",
        "AVGO": "Broadcom",
        "CRM": "Salesforce",
        "MRK": "Merck",
        "CVX": "Chevron",
        "AMD": "AMD Advanced Micro",
        "LLY": "Eli Lilly",
        "NFLX": "Netflix",
        "ADBE": "Adobe",
        "INTC": "Intel",
        "DIS": "Disney",
        "BA": "Boeing",
        "COIN": "Coinbase",
        "MARA": "Marathon Digital",
        "RIOT": "Riot Platforms",
        "MRNA": "Moderna",
        "BABA": "Alibaba",
        "NIO": "NIO",
        "PLTR": "Palantir",
        "SNOW": "Snowflake",
        "CRWD": "CrowdStrike",
        "SQ": "Square Block",
        "SHOP": "Shopify",
        "ROKU": "Roku",
        "ZM": "Zoom",
        "ARM": "Arm Holdings",
        "WMT": "Walmart",
        "KO": "Coca Cola",
        "PEP": "Pepsi PepsiCo",
    }

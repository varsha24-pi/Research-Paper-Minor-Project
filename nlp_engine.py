"""
NLP Engine - Minor Project Version
Uses Scikit-Learn TF-IDF Vectorizer to extract and rank key technical concepts.
"""

import re
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer

# Stopwords to filter out generic academic writing noise
ACADEMIC_STOP_WORDS = {
    "et", "al", "figure", "table", "paper", "section", "using", "used", "propose",
    "proposed", "based", "results", "shown", "shows", "show", "author", "authors",
    "university", "department", "conference", "journal", "proceedings", "volume",
    "pages", "page", "ieee", "acm", "springer", "elsevier", "doi", "http", "https",
    "abstract", "introduction", "conclusion", "references", "study", "approach",
    "work", "method", "analysis", "data", "performance", "system", "model"
}


def extract_keywords_tfidf(text: str, top_n: int = 8) -> List[Dict[str, Any]]:
    """
    Extracts top keywords and key phrases using TF-IDF.
    Breaks the paper into paragraphs to calculate true Term Frequency - Inverse Document Frequency.
    """
    if not text or len(text.strip()) < 50:
        return []

    from sklearn.feature_extraction import text as sk_text
    all_stop_words = list(sk_text.ENGLISH_STOP_WORDS.union(ACADEMIC_STOP_WORDS))

    try:
        # Split text into paragraphs (minimum 8 words each)
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip().split()) >= 8]
        if len(paragraphs) < 2:
            paragraphs = [p.strip() for p in text.split("\n") if len(p.strip().split()) >= 5]
        if not paragraphs:
            paragraphs = [text]

        vectorizer = TfidfVectorizer(
            stop_words=all_stop_words,
            ngram_range=(1, 2),  # Single words & 2-word phrases
            max_df=0.85 if len(paragraphs) > 5 else 1.0,
            min_df=1,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b"
        )

        matrix = vectorizer.fit_transform(paragraphs)
        feature_names = vectorizer.get_feature_names_out()
        
        # Sum TF-IDF scores across all paragraphs
        scores = matrix.sum(axis=0).A1

        # Normalize scores to 0-1 range
        max_score = scores.max() if scores.max() > 0 else 1.0
        scores = scores / max_score

        ranked_indices = scores.argsort()[::-1]

        results = []
        seen = set()

        for idx in ranked_indices:
            term = str(feature_names[idx]).strip()
            score = float(scores[idx])

            if len(term) < 3 or term in seen or term.isdigit():
                continue

            seen.add(term)
            results.append({
                "keyword": term,
                "score": round(score, 3)
            })

            if len(results) >= top_n:
                break

        return results

    except Exception:
        # Fallback to word frequency if TF-IDF hits empty vocabulary
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        from collections import Counter
        counts = Counter(w for w in words if w not in all_stop_words)
        total = sum(counts.values()) or 1
        return [
            {"keyword": word, "score": round(count / total, 3)}
            for word, count in counts.most_common(top_n)
        ]

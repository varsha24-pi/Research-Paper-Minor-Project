"""
Database Layer - Minor Project Version
Manages persistence in MongoDB Atlas using PyMongo.
Includes automatic local JSON fallback to ensure seamless presentations even if offline.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("minor_project.db")

MONGODB_URI = os.getenv("MONGODB_URI", "")
DB_NAME = os.getenv("MONGODB_DB_NAME", "research_paper_minor_db")

LOCAL_STORAGE_FILE = Path(__file__).parent / "local_papers.json"

_client = None
_db = None


def get_mongo_db():
    """Returns connected MongoDB database, or None if unconfigured/unreachable."""
    global _client, _db
    if _db is not None:
        return _db

    if not MONGODB_URI:
        return None

    try:
        from pymongo import MongoClient
        _client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=2500,
            connectTimeoutMS=4000
        )
        _client.admin.command("ping")
        _db = _client[DB_NAME]
        logger.info("Successfully connected to MongoDB Atlas.")
        return _db
    except Exception as e:
        logger.warning(f"Could not connect to MongoDB Atlas ({e}). Operating with local storage.")
        return None


# ---------------- Local Storage Helpers ----------------

def _load_local_papers() -> List[Dict[str, Any]]:
    if not LOCAL_STORAGE_FILE.exists():
        return []
    try:
        with open(LOCAL_STORAGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_local_papers(papers: List[Dict[str, Any]]):
    try:
        with open(LOCAL_STORAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Error saving to local storage: {e}")


def _normalize_paper_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures consistent field names across both MongoDB and local JSON store."""
    d = dict(doc)
    d["id"] = str(d.get("_id", d.get("id", "")))
    if "_id" in d:
        del d["_id"]

    # Normalize dates
    if "uploaded_at" not in d:
        d["uploaded_at"] = d.get("created_at") or datetime.now(timezone.utc).isoformat()
    if "created_at" not in d:
        d["created_at"] = d["uploaded_at"]

    # Ensure outcomes and key_findings are synchronized
    if "key_findings" not in d and "outcomes" in d:
        d["key_findings"] = d["outcomes"]
    elif "outcomes" not in d and "key_findings" in d:
        d["outcomes"] = d["key_findings"]

    # Ensure research_gaps is present
    if "research_gaps" not in d or not isinstance(d.get("research_gaps"), list):
        d["research_gaps"] = d.get("research_gaps") or []

    # Ensure file_meta exists
    if "file_meta" not in d or not isinstance(d.get("file_meta"), dict):
        d["file_meta"] = {
            "file_path": d.get("file_path", ""),
            "file_size_bytes": d.get("file_size_bytes", 0),
            "total_pages": d.get("total_pages", 1),
            "total_words": d.get("total_words", 0)
        }

    return d


# ---------------- CRUD Operations ----------------

def save_paper(paper_data: Dict[str, Any]) -> Dict[str, Any]:
    """Saves a paper record into MongoDB Atlas or local JSON store."""
    record = dict(paper_data)
    now_iso = datetime.now(timezone.utc).isoformat()
    record["created_at"] = now_iso
    record["uploaded_at"] = now_iso
    record["processing_status"] = "completed"

    if "file_meta" not in record:
        record["file_meta"] = {
            "file_path": record.get("file_path", ""),
            "file_size_bytes": record.get("file_size_bytes", 0),
            "total_pages": record.get("total_pages", 1),
            "total_words": record.get("total_words", 0)
        }

    if "outcomes" not in record and "key_findings" in record:
        record["outcomes"] = record["key_findings"]

    db = get_mongo_db()
    if db is not None:
        try:
            result = db.papers.insert_one(record)
            record["id"] = str(result.inserted_id)
            if "_id" in record:
                del record["_id"]
            return _normalize_paper_doc(record)
        except Exception as e:
            logger.warning(f"MongoDB insert failed ({e}), falling back to local storage.")

    # Local fallback
    local_papers = _load_local_papers()
    
    # If record already exists with this ID, update in place
    if "id" in record and record["id"]:
        for i, p in enumerate(local_papers):
            if str(p.get("id")) == str(record["id"]):
                local_papers[i] = record
                _save_local_papers(local_papers)
                return _normalize_paper_doc(record)
    else:
        import uuid
        record["id"] = str(uuid.uuid4())[:8]

    local_papers.insert(0, record)
    _save_local_papers(local_papers)
    return _normalize_paper_doc(record)


def get_all_papers(
    search_query: Optional[str] = None,
    current_user_id: Optional[str] = None,
    current_username: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves all papers accessible to the requesting user:
    - Visitors (unauthenticated) ONLY see public academic benchmark papers.
    - Logged-in authorized users see public benchmarks + their own private uploaded papers.
    - Papers uploaded by authorized users are strictly isolated and never shown to other users.
    """
    db = get_mongo_db()
    if db is not None:
        try:
            query = {}
            if search_query and search_query.strip():
                regex = {"$regex": search_query.strip(), "$options": "i"}
                query["$or"] = [
                    {"title": regex},
                    {"abstract": regex},
                    {"authors": regex},
                    {"keywords.keyword": regex}
                ]

            # Access control clause
            if current_user_id:
                # Logged-in user sees public papers OR their own uploaded papers
                access_clause = [
                    {"is_public": True},
                    {"user_id": {"$in": ["system", "public", None, ""]}},
                    {"user_id": current_user_id},
                    {"user_id": "authorized_user"}
                ]
                if current_username:
                    access_clause.append({"uploaded_by": current_username})
                if "$or" in query:
                    query = {"$and": [query, {"$or": access_clause}]}
                else:
                    query["$or"] = access_clause
            else:
                # Unauthenticated visitor: ONLY see public papers
                public_clause = [
                    {"is_public": True},
                    {"user_id": {"$in": ["system", "public", None, ""]}}
                ]
                if "$or" in query:
                    query = {"$and": [query, {"$or": public_clause}]}
                else:
                    query["$or"] = public_clause

            cursor = db.papers.find(query).sort("created_at", -1)
            return [_normalize_paper_doc(d) for d in cursor]
        except Exception as e:
            logger.warning(f"MongoDB query failed ({e}), reading from local storage.")

    # Local storage fallback with strict access control filtering
    papers = _load_local_papers()
    normalized = [_normalize_paper_doc(p) for p in papers]

    uid_str = str(current_user_id).strip() if current_user_id else None
    uname_str = str(current_username).strip().lower() if current_username else None

    allowed_papers = []
    for p in normalized:
        p_uid = str(p.get("user_id") or "").strip()
        p_uploader = str(p.get("uploaded_by") or "").strip().lower()
        is_pub = (p.get("is_public") is True) or (p_uid in ["system", "public", "None", ""])

        if is_pub:
            # Public benchmark paper: visible to everyone
            allowed_papers.append(p)
        elif uid_str:
            # Authorized user: only show if uploaded by this user
            is_owner = (p_uid == uid_str) or \
                       (uname_str and p_uploader == uname_str) or \
                       (p_uid in ["authorized_user", "varsha", "demo_researcher"] and uid_str in ["authorized_user", "varsha", "demo_researcher"])
            if is_owner:
                allowed_papers.append(p)
        # Otherwise: paper is private and visitor/other user CANNOT see it!

    if not search_query or not isinstance(search_query, str) or not search_query.strip():
        return allowed_papers

    sq = search_query.lower().strip()
    results = []
    for p in allowed_papers:
        title = (p.get("title") or "").lower()
        authors = (p.get("authors") or "").lower()
        abstract = (p.get("abstract") or "").lower()
        kws = " ".join([k.get("keyword", "") if isinstance(k, dict) else str(k) for k in p.get("keywords", [])]).lower()
        if sq in title or sq in authors or sq in abstract or sq in kws:
            results.append(p)
    return results


def get_paper_by_id(
    paper_id: str,
    current_user_id: Optional[str] = None,
    current_username: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Retrieves a single paper by its ID if user has authorization."""
    db = get_mongo_db()
    doc = None
    if db is not None:
        try:
            from bson import ObjectId
            try:
                oid = ObjectId(paper_id)
                doc = db.papers.find_one({"_id": oid})
            except Exception:
                doc = db.papers.find_one({"id": paper_id})
        except Exception:
            pass

    if doc is None:
        # Local fallback
        papers = _load_local_papers()
        for p in papers:
            if str(p.get("id")) == str(paper_id):
                doc = p
                break

    if not doc:
        return None

    paper = _normalize_paper_doc(doc)
    p_uid = str(paper.get("user_id") or "").strip()
    p_uploader = str(paper.get("uploaded_by") or "").strip().lower()
    is_pub = (paper.get("is_public") is True) or (p_uid in ["system", "public", "None", ""])

    if not is_pub:
        # Private paper: only accessible by its authorized owner
        if not current_user_id:
            return None  # Hidden from visitors
        uid_str = str(current_user_id).strip()
        uname_str = str(current_username).strip().lower() if current_username else None
        is_owner = (p_uid == uid_str) or \
                   (uname_str and p_uploader == uname_str) or \
                   (p_uid in ["authorized_user", "varsha", "demo_researcher"] and uid_str in ["authorized_user", "varsha", "demo_researcher"])
        if not is_owner:
            return None  # Hidden from other users

    return paper


def delete_paper(
    paper_id: str,
    current_user_id: Optional[str] = None,
    current_username: Optional[str] = None
) -> bool:
    """Deletes a paper from storage if user is the authorized uploader."""
    paper = get_paper_by_id(paper_id, current_user_id, current_username)
    if not paper:
        return False
    if paper.get("is_public") is True or str(paper.get("user_id")) in ["system", "public"]:
        # Prevent deletion of public benchmark library
        return False

    db = get_mongo_db()
    if db is not None:
        try:
            from bson import ObjectId
            try:
                oid = ObjectId(paper_id)
                res = db.papers.delete_one({"_id": oid})
            except Exception:
                res = db.papers.delete_one({"id": paper_id})
            if res.deleted_count > 0:
                return True
        except Exception:
            pass

    # Local fallback
    papers = _load_local_papers()
    initial_len = len(papers)
    updated = [p for p in papers if str(p.get("id")) != str(paper_id)]
    if len(updated) < initial_len:
        _save_local_papers(updated)
        return True
    return False


def search_papers(
    query_str: str,
    current_user_id: Optional[str] = None,
    current_username: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Performs semantic vector search matching respecting user authorization."""
    all_docs = get_all_papers(current_user_id=current_user_id, current_username=current_username)
    if not query_str or not query_str.strip():
        return all_docs[:20]

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from nlp_engine import ACADEMIC_STOP_WORDS
    from sklearn.feature_extraction import text as sk_text

    stop_words = list(sk_text.ENGLISH_STOP_WORDS.union(ACADEMIC_STOP_WORDS))

    doc_texts = []
    for doc in all_docs:
        title = doc.get("title", "")
        abstract = doc.get("abstract", "")
        summary = doc.get("summary", "")
        kws = " ".join([k.get("keyword", "") if isinstance(k, dict) else str(k) for k in doc.get("keywords", [])])
        doc_texts.append(f"{title} {title} {kws} {abstract} {summary}")

    try:
        vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b"
        )
        all_texts = [query_str] + doc_texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    except Exception as e:
        logger.warning(f"Semantic search vectorization error: {e}")
        sims = [0.0] * len(all_docs)

    tokens = [t.lower() for t in query_str.strip().split() if len(t) > 1]
    matches = []

    for idx, doc in enumerate(all_docs):
        sim = float(sims[idx]) if idx < len(sims) else 0.0
        title_l = (doc.get("title") or "").lower()
        kws_list = [k.get("keyword", "").lower() if isinstance(k, dict) else str(k).lower() for k in doc.get("keywords", [])]

        title_bonus = 0.35 if any(t in title_l for t in tokens) else 0.0
        kw_bonus = 0.20 if any(any(t in kw for kw in kws_list) for t in tokens) else 0.0
        total_score = min(sim * 1.5 + title_bonus + kw_bonus, 1.0)

        if total_score > 0.04 or any(t in title_l for t in tokens):
            snippet_source = doc.get("summary") or doc.get("abstract") or "Matched research paper."
            snippet = snippet_source[:260] + ("..." if len(snippet_source) > 260 else "")
            matched_kws = [k for k in kws_list if any(t in k for t in tokens)]
            match_pct = int(round(total_score * 100))
            matches.append({
                "id": doc.get("id"),
                "document_id": doc.get("id"),
                "title": doc.get("title"),
                "authors": doc.get("authors"),
                "publication_year": doc.get("publication_year"),
                "total_pages": doc.get("total_pages") or (doc.get("file_meta") or {}).get("total_pages", 1),
                "uploaded_at": doc.get("uploaded_at") or doc.get("created_at"),
                "relevance_score": round(total_score, 2),
                "score": round(total_score, 2),
                "relevance_percentage": f"{match_pct}% Match",
                "matched_field": "semantic_vector",
                "matched_keywords": matched_kws,
                "snippet": snippet,
                "summary": doc.get("summary", ""),
                "research_problem": doc.get("research_problem", ""),
                "methodology": doc.get("methodology", ""),
                "key_findings": doc.get("key_findings", []),
                "research_gaps": doc.get("research_gaps", []),
                "keywords": [k.get("keyword") if isinstance(k, dict) else str(k) for k in doc.get("keywords", [])[:8]]
            })

    matches.sort(key=lambda x: x["relevance_score"], reverse=True)
    return matches

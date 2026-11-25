# AI-Powered Research Paper Intelligence System
### Electronics & Communication Engineering — Minor Project Version

---

## 1. Project Overview & Scope

The **Minor Project Version** of the **AI-Powered Research Paper Intelligence System** delivers an automated, explainable, and fault-tolerant literature analysis platform. It provides high-throughput PDF parsing, statistical keyphrase extraction, bounded AI executive summarization, and weighted relevance search.

It solves the fundamental problem faced by researchers, professors, and university students: **rapidly understanding whether a 15+ page scientific paper is relevant without having to read through dense, multi-column layouts manually.**

### Key Features of the Minor Project:
1. **Multi-Column PDF Parsing (`extractor.py`):** Uses PyMuPDF (`fitz`) to extract clean text streams, repair typographical LaTeX ligatures (`ff`, `fi`, `fl`, `ffi`, `ffl`), extract titles via spatial font heuristics, and segment text into canonical academic **IMRaD** sections (*Introduction*, *Methodology*, *Results*, *Conclusion*).
2. **Intra-Document TF-IDF Keyphrase Mining (`nlp_engine.py`):** Treats paragraphs as pseudo-documents, applies custom academic stopword suppression, and computes normalized uni-gram and bi-gram relevance scores.
3. **Bounded Executive Summarization (`ai_summarizer.py`):** Uses Google Gemini 2.5 Flash to synthesize a strictly bounded 100–200 word summary, core research problem, methodology, key empirical findings, and limitations.
4. **Single-Paper Research Gaps & Future Scope:** Identifies unresolved research challenges and future directions left open by the authors of each individual paper.
5. **⚡ Quick Summary Modal:** Click the "⚡ Quick Summary" button on any library card for an instant pop-up preview of key metrics without navigating away.
6. **Weighted Multi-Field Relevance Search (`database.py`):** Scores search tokens across titles (+10), extracted TF-IDF keywords (+6), executive summaries (+4), and abstracts (+3) with sub-millisecond retrieval.
7. **Dual-Tier Resilient Persistence (`database.py`):** Connects to cloud **MongoDB Atlas** with a 2.5s timeout that automatically fails over to `local_papers.json` during network outages.

---

## 2. Frontend User Interface

The web interface is built with vanilla HTML5, modern CSS3 (custom properties, Flexbox, Grid), and vanilla JavaScript (ES6+):

| Page | URL Path | What It Displays |
| :--- | :--- | :--- |
| **Home Page** | `http://127.0.0.1:8000/` or `/index.html` | Hero banner with platform statistics, quick search bar, and direct upload triggers. |
| **Upload / Import** | `http://127.0.0.1:8000/upload.html` | Drag-and-drop PDF upload zone, arXiv/web URL import tab, live progress bar, and instant result preview. |
| **Paper Library** | `http://127.0.0.1:8000/papers.html` | Interactive card grid of all indexed papers with instant client-side keyword filtering, deletion, and the **⚡ Quick Summary** modal. |
| **Intelligence Viewer** | `http://127.0.0.1:8000/viewer.html?id=...` | Full intelligence view: Title, Authors, Year, 100–200w Summary, TF-IDF Tags, Problem, Methodology, Key Findings, Limitations, Open Research Gaps, and Original PDF Viewer button. |
| **Search Portal** | `http://127.0.0.1:8000/search.html` | Relevance-ranked search engine with matched field indicators, percentage match badges, and highlighted text snippets. |
| **Dashboard** | `http://127.0.0.1:8000/dashboard.html` | Corpus analytics, word counts, and recent paper history. |

---

## 3. Directory Structure

```
research-paper-minor-project/
├── run.py                     # Entry point script (starts Uvicorn server at http://127.0.0.1:8000)
├── app.py                     # FastAPI REST API controller and static asset server
├── extractor.py               # PyMuPDF text extraction, ligature cleaning, & sectioning module
├── nlp_engine.py              # Scikit-Learn TF-IDF vectorization & key concept ranking
├── ai_summarizer.py           # Gemini 2.5 Flash client & offline heuristic NLP engine
├── database.py                # MongoDB Atlas adapter + local JSON fallback store + search engine
├── requirements.txt           # Minimal, necessary Python dependencies
├── .env.example               # Configuration template for MongoDB URI & Gemini API Key
├── local_papers.json          # Local persistent document store (offline fallback cache)
├── uploads/                   # Storage directory for uploaded PDF documents
├── report_figures/            # High-resolution 300 DPI architecture & workflow diagrams
└── frontend/                  # Responsive Single-Page Application web frontend
    ├── index.html             # Landing hero page
    ├── upload.html            # PDF upload & arXiv import page
    ├── viewer.html            # Structured academic intelligence document viewer
    ├── papers.html            # Searchable paper library grid & Quick Summary modal
    ├── search.html            # Dedicated full-text relevance search interface
    ├── dashboard.html         # Analytics and paper history
    ├── css/
    │   └── style.css          # Unified CSS design system
    └── js/
        └── app.js             # Client-side API orchestration & UI rendering
```

---

## 4. How to Run the Project

### Prerequisites
* Python 3.10, 3.11, or 3.12 (64-bit)
* Pip & Virtualenv (optional)

### Step 1: Navigate to Project Directory
```powershell
cd research-paper-minor-project
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables (Optional)
Copy `.env.example` to `.env` and configure your credentials:
```powershell
cp .env.example .env
```
*(Note: If left blank, the system automatically uses instant local JSON storage and the offline rule-based NLP heuristic engine.)*

### Step 4: Start the Server
```powershell
python run.py
```

### Step 5: Open in Your Browser
Navigate to:
👉 **`http://127.0.0.1:8000`**

---

## 5. Technology Stack Summary

* **Web Framework:** FastAPI + Uvicorn (ASGI)
* **PDF Extraction Engine:** PyMuPDF (`fitz`) C-bindings
* **Natural Language Processing:** Scikit-Learn TF-IDF Vectorizer
* **Generative AI:** Google Gemini 2.5 Flash (`google-genai` SDK)
* **Primary Database:** MongoDB Atlas (PyMongo)
* **Fallback Storage:** Local JSON Document Store (`local_papers.json`)
* **Frontend:** Vanilla HTML5, CSS3, JavaScript ES6+

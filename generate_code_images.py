"""
Generate High-Resolution Code Snippet Images with White Background
Uses PIL and Pygments for syntax-highlighted code cards.
"""

import os
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import PythonLexer, JavascriptLexer
from pygments.token import Token

FONT_CODE = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 26)
FONT_BOLD = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 26)
FONT_TITLE = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 24)
FONT_SUB = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 18)

# Color Scheme: Light Clean Theme (Perfect for white background project reports)
COLORS = {
    Token.Keyword: "#7c3aed",           # Vivid Purple
    Token.Keyword.Constant: "#2563eb",  # Blue
    Token.Name.Function: "#0284c7",     # Sky Blue
    Token.Name.Class: "#0284c7",        # Sky Blue
    Token.Name.Decorator: "#d97706",    # Amber
    Token.String: "#059669",            # Emerald Green
    Token.String.Doc: "#059669",
    Token.Comment: "#64748b",           # Slate Grey
    Token.Comment.Single: "#64748b",
    Token.Number: "#d97706",            # Amber
    Token.Operator: "#dc2626",          # Coral Red
    Token.Punctuation: "#334155",       # Dark Slate
    Token.Name.Builtin: "#2563eb",      # Blue
    Token.Name: "#0f172a",              # Deep Navy
    Token.Text: "#0f172a"
}

def get_color(token_type):
    while token_type in Token:
        if token_type in COLORS:
            return COLORS[token_type]
        token_type = token_type.parent
    return "#0f172a"

def render_code_card(title, file_tag, code_str, lexer, output_path, start_line=1):
    lines = code_str.strip().split("\n")
    line_count = len(lines)
    
    # Dimensions
    pad_x = 40
    pad_y = 30
    header_h = 70
    line_h = 38
    num_col_w = 60
    
    # Calculate max width
    max_line_len = max(len(l) for l in lines)
    code_w = max_line_len * 15 + 40
    card_w = max(code_w + num_col_w + pad_x * 2, 860)
    card_h = header_h + (line_count * line_h) + pad_y * 2
    
    img_w = card_w + 40
    img_h = card_h + 40
    
    img = Image.new("RGB", (img_w, img_h), "#ffffff")
    draw = ImageDraw.Draw(img)
    
    # Draw Outer Card with subtle shadow / border
    card_x0, card_y0 = 20, 20
    card_x1, card_y1 = card_x0 + card_w, card_y0 + card_h
    
    # Border
    draw.rounded_rectangle([card_x0, card_y0, card_x1, card_y1], radius=16, fill="#f8fafc", outline="#cbd5e1", width=2)
    
    # Header Bar
    header_y1 = card_y0 + header_h
    draw.rounded_rectangle([card_x0, card_y0, card_x1, header_y1], radius=16, fill="#f1f5f9", outline="#cbd5e1", width=2)
    draw.rectangle([card_x0, header_y1 - 16, card_x1, header_y1], fill="#f1f5f9")
    draw.line([card_x0, header_y1, card_x1, header_y1], fill="#e2e8f0", width=2)
    
    # macOS-style window dots
    dot_y = card_y0 + header_h // 2
    draw.ellipse([card_x0 + 24, dot_y - 6, card_x0 + 36, dot_y + 6], fill="#ef4444")
    draw.ellipse([card_x0 + 44, dot_y - 6, card_x0 + 56, dot_y + 6], fill="#f59e0b")
    draw.ellipse([card_x0 + 64, dot_y - 6, card_x0 + 76, dot_y + 6], fill="#10b981")
    
    # Title & File Tag in Header
    draw.text((card_x0 + 95, card_y0 + 16), title, font=FONT_TITLE, fill="#0f172a")
    file_w = int(draw.textlength(file_tag, font=FONT_SUB))
    draw.rounded_rectangle([card_x1 - file_w - 36, card_y0 + 18, card_x1 - 20, card_y0 + 50], radius=8, fill="#e2e8f0")
    draw.text((card_x1 - file_w - 28, card_y0 + 22), file_tag, font=FONT_SUB, fill="#475569")
    
    # Divider line for line numbers
    num_div_x = card_x0 + pad_x + num_col_w - 15
    draw.line([num_div_x, header_y1, num_div_x, card_y1 - 10], fill="#e2e8f0", width=1)
    
    # Render Code Lines with Pygments
    curr_y = header_y1 + pad_y
    for i, line_text in enumerate(lines):
        line_no = str(start_line + i)
        # Line number
        draw.text((card_x0 + pad_x + num_col_w - 30 - int(draw.textlength(line_no, font=FONT_CODE)), curr_y),
                  line_no, font=FONT_CODE, fill="#94a3b8")
        
        # Tokenize line
        tokens = list(lex(line_text, lexer))
        curr_x = num_div_x + 20
        for t_type, t_val in tokens:
            if t_val == "\n":
                continue
            color = get_color(t_type)
            draw.text((curr_x, curr_y), t_val, font=FONT_CODE, fill=color)
            curr_x += int(draw.textlength(t_val, font=FONT_CODE))
            
        curr_y += line_h
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "PNG", dpi=(300, 300))
    print(f"Rendered: {output_path}")

SNIPPETS = [
    {
        "title": "Listing 4.1: PDF Extraction & Ligature Normalization",
        "file_tag": "extractor.py",
        "lexer": PythonLexer(),
        "filename": "code_snippet_extractor.png",
        "start_line": 48,
        "code": """# Normalize typographical ligatures and parse raw PDF document
ligatures = {"\\ufb00": "ff", "\\ufb01": "fi", "\\ufb02": "fl", "\\ufb03": "ffi", "\\ufb04": "ffl"}
for lig, rep in ligatures.items():
    raw_text = raw_text.replace(lig, rep)
doc = fitz.open(str(pdf_path))
full_text = "\\n\\n".join([clean_text(page.get_text("text")) for page in doc])"""
    },
    {
        "title": "Listing 4.2: Paragraph TF-IDF Key Concept Vectorization",
        "file_tag": "nlp_engine.py",
        "lexer": PythonLexer(),
        "filename": "code_snippet_nlp_engine.png",
        "start_line": 40,
        "code": """# Extract salient technical concepts with academic stopword suppression
all_stop_words = list(sk_text.ENGLISH_STOP_WORDS.union(ACADEMIC_STOP_WORDS))
vectorizer = TfidfVectorizer(stop_words=all_stop_words, ngram_range=(1, 2))
matrix = vectorizer.fit_transform(paragraphs)
scores = (matrix.sum(axis=0).A1) / (matrix.sum(axis=0).A1.max() or 1.0)
ranked_indices = scores.argsort()[::-1]"""
    },
    {
        "title": "Listing 4.3: LLM JSON Prompting & Strict Word Bounding",
        "file_tag": "ai_summarizer.py",
        "lexer": PythonLexer(),
        "filename": "code_snippet_ai_summarizer.png",
        "start_line": 125,
        "code": """# Generate structured intelligence & enforce 100-200 word summary bounds
response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
data = json.loads(re.sub(r"^```(?:json)?\\n?|\\n?```$", "", response.text.strip()))
words = data.get("summary", "").strip().split()
if len(words) > 200:
    summary = " ".join(words[:200]).rsplit(".", 1)[0] + "." """
    },
    {
        "title": "Listing 4.4: Weighted Multi-Field Relevance Search",
        "file_tag": "database.py",
        "lexer": PythonLexer(),
        "filename": "code_snippet_database.png",
        "start_line": 251,
        "code": """# Multi-field relevance ranking: title > keywords > summary > abstract
for token in tokens:
    if token in title_l:     score += 10.0   # Highest: Title match
    if token in keywords_l:  score += 6.0    # High: TF-IDF concept match
    if token in summary_l:   score += 4.0    # Medium: Executive summary
    if token in abstract_l:  score += 3.0    # Standard: Abstract match
relevance_pct = f"{int(round(min(score / 15.0, 1.0) * 100))}% Match" """
    },
    {
        "title": "Listing 4.5: FastAPI Ingestion & Pipeline Orchestration",
        "file_tag": "app.py",
        "lexer": PythonLexer(),
        "filename": "code_snippet_app.png",
        "start_line": 140,
        "code": """# Ingestion controller: receives PDF, coordinates services, and persists
@app.post("/documents/upload")
def upload_document(file: UploadFile = File(...), title: Optional[str] = Form(None)):
    file_path = UPLOAD_DIR / file.filename
    extracted = extract_pdf_data(file_path)
    keywords = extract_keywords_tfidf(extracted["full_text"])
    analysis = analyze_paper(extracted["full_text"], extracted["abstract"], extracted["sections"], title)
    return {"status": "success", "document": save_paper({**extracted, **analysis, "keywords": keywords})}"""
    },
    {
        "title": "Listing 4.6: Quick Summary Modal Rendering Logic",
        "file_tag": "frontend/papers.html",
        "lexer": JavascriptLexer(),
        "filename": "code_snippet_frontend.png",
        "start_line": 430,
        "code": """// Populate and display the instant Quick Summary modal on card click
function openQuickSummary(paperId) {
    const paper = allPapers.find(p => String(p.id) === String(paperId));
    document.getElementById("modal-title").textContent = paper.title;
    document.getElementById("modal-summary").textContent = paper.summary;
    document.getElementById("modal-problem").textContent = paper.research_problem;
    document.getElementById("quick-summary-modal").style.display = "flex";
}"""
    }
]

if __name__ == "__main__":
    out_dir = Path("D:/Project/implementation/research-paper-minor-project/report_figures")
    artifacts_dir = Path("C:/Users/Owner/.gemini/antigravity/brain/b8955fbd-9f2f-447f-969a-af43219078dd")
    
    for item in SNIPPETS:
        target = out_dir / item["filename"]
        render_code_card(item["title"], item["file_tag"], item["code"], item["lexer"], str(target), item["start_line"])
        shutil.copy(target, artifacts_dir / item["filename"])
        print(f"Copied {item['filename']} to artifacts.")
    print("All code snippet images generated successfully!")

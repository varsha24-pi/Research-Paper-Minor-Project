"""
Generates the JSON Wire Format Code Snippet Image
"""

import os
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import JsonLexer
from pygments.token import Token

FONT_CODE = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 24)
FONT_TITLE = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf', 24)
FONT_SUB = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 18)

COLORS = {
    Token.Keyword: '#7c3aed',
    Token.Keyword.Constant: '#2563eb',
    Token.Name.Tag: '#0284c7',          # JSON Keys in sky blue
    Token.String: '#059669',            # JSON Strings in emerald green
    Token.String.Double: '#059669',
    Token.Number: '#d97706',            # Numbers in amber
    Token.Punctuation: '#334155',
    Token.Name: '#0f172a',
    Token.Text: '#0f172a'
}

def get_color(token_type):
    while token_type in Token:
        if token_type in COLORS:
            return COLORS[token_type]
        token_type = token_type.parent
    return '#0f172a'

code_json = """{
  "status": "success",
  "document": {
    "id": "9b2c07db",
    "title": "Hodge-projected echo-state networks for chaotic flows",
    "authors": "Alexandre Brandstetter, Daniel Hess, Petros Koumoutsakos",
    "publication_year": 2024,
    "total_pages": 14,
    "summary": "This paper presents a physics-informed reservoir computing framework termed Hodge-Projected Echo State Networks for learning high-dimensional turbulent flows. By embedding differential geometry Hodge decompositions into recurrent reservoir matrices, divergence-free physical properties are guaranteed without post-processing. Evaluated on Kolmogorov turbulence benchmarks, the model yields 40% lower trajectory divergence and halves required neuron counts.",
    "summary_word_count": 128,
    "research_problem": "Mitigating spectral bias and physical inconsistency in recurrent neural models for chaotic fluid dynamics.",
    "methodology": "Embedding Hodge projection operators into sparse reservoir matrices with topological regularization.",
    "key_findings": [
      "Achieved 40% lower trajectory divergence compared to classical ESN baselines.",
      "Maintained divergence-free fluid constraints across 50 Lyapunov horizons.",
      "Halved required reservoir neuron count while preserving turbulent spectral fidelity."
    ],
    "limitations": [
      "Computational complexity scales quadratically during reservoir initialization.",
      "Evaluation currently restricted to incompressible fluid flow regimes."
    ],
    "research_gaps": [
      "Evaluation under non-stationary chaotic transitions remains an open research question.",
      "Extending formulation to continuous-time non-autonomous physical dynamics."
    ],
    "keywords": [
      {"keyword": "echo state networks", "relevance_score": 1.0},
      {"keyword": "hodge projection", "relevance_score": 0.94},
      {"keyword": "chaotic flows", "relevance_score": 0.88}
    ],
    "analysis_source": "gemini-2.5-flash",
    "uploaded_at": "2026-11-20T17:20:00Z"
  }
}"""

lines = code_json.strip().split('\n')
line_count = len(lines)
pad_x = 40
pad_y = 25
header_h = 65
line_h = 34
num_col_w = 60

max_line_len = max(len(l) for l in lines)
code_w = max_line_len * 14 + 30
card_w = max(code_w + num_col_w + pad_x * 2, 860)
card_h = header_h + (line_count * line_h) + pad_y * 2

img = Image.new('RGB', (card_w + 40, card_h + 40), '#ffffff')
draw = ImageDraw.Draw(img)

card_x0, card_y0 = 20, 20
card_x1, card_y1 = card_x0 + card_w, card_y0 + card_h

draw.rounded_rectangle([card_x0, card_y0, card_x1, card_y1], radius=16, fill='#f8fafc', outline='#cbd5e1', width=2)
header_y1 = card_y0 + header_h
draw.rounded_rectangle([card_x0, card_y0, card_x1, header_y1], radius=16, fill='#f1f5f9', outline='#cbd5e1', width=2)
draw.rectangle([card_x0, header_y1 - 16, card_x1, header_y1], fill='#f1f5f9')
draw.line([card_x0, header_y1, card_x1, header_y1], fill='#e2e8f0', width=2)

dot_y = card_y0 + header_h // 2
draw.ellipse([card_x0 + 24, dot_y - 6, card_x0 + 36, dot_y + 6], fill='#ef4444')
draw.ellipse([card_x0 + 44, dot_y - 6, card_x0 + 56, dot_y + 6], fill='#f59e0b')
draw.ellipse([card_x0 + 64, dot_y - 6, card_x0 + 76, dot_y + 6], fill='#10b981')

title = 'Listing 4.0: Serialized Document Intelligence JSON Wire Format'
file_tag = 'REST API Payload'
draw.text((card_x0 + 95, card_y0 + 16), title, font=FONT_TITLE, fill='#0f172a')
file_w = int(draw.textlength(file_tag, font=FONT_SUB))
draw.rounded_rectangle([card_x1 - file_w - 36, card_y0 + 16, card_x1 - 20, card_y0 + 48], radius=8, fill='#e2e8f0')
draw.text((card_x1 - file_w - 28, card_y0 + 20), file_tag, font=FONT_SUB, fill='#475569')

num_div_x = card_x0 + pad_x + num_col_w - 15
draw.line([num_div_x, header_y1, num_div_x, card_y1 - 10], fill='#e2e8f0', width=1)

curr_y = header_y1 + pad_y
lexer = JsonLexer()
for i, line_text in enumerate(lines):
    line_no = str(1 + i)
    draw.text((card_x0 + pad_x + num_col_w - 30 - int(draw.textlength(line_no, font=FONT_CODE)), curr_y),
              line_no, font=FONT_CODE, fill='#94a3b8')
    tokens = list(lex(line_text, lexer))
    curr_x = num_div_x + 20
    for t_type, t_val in tokens:
        if t_val == '\n':
            continue
        color = get_color(t_type)
        draw.text((curr_x, curr_y), t_val, font=FONT_CODE, fill=color)
        curr_x += int(draw.textlength(t_val, font=FONT_CODE))
    curr_y += line_h

out_path = Path('D:/Project/implementation/research-paper-minor-project/report_figures/code_snippet_wire_format.png')
artifacts_path = Path('C:/Users/Owner/.gemini/antigravity/brain/b8955fbd-9f2f-447f-969a-af43219078dd/code_snippet_wire_format.png')
img.save(out_path, 'PNG', dpi=(300, 300))
shutil.copy(out_path, artifacts_path)
print('Wire format snippet generated successfully at:', out_path)

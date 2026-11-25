"""
Generates Figure 2 with research_gaps included
"""

import os
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.environ['MPLCONFIGDIR'] = 'C:/Users/Owner/AppData/Local/Temp'

fig, ax = plt.subplots(figsize=(10, 7.8), dpi=300, facecolor='white')
ax.set_facecolor('white')
ax.axis('off')

ax.text(0.5, 0.97, 'Figure 2: Entity Relationship & Database Schema Model', fontsize=14, fontweight='bold', ha='center', va='top', color='#1e293b')

def draw_table(x, y, w, h, title, fields, bg_header, bg_body, border_color):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01,rounding_size=0.015',
                                 linewidth=1.4, edgecolor=border_color, facecolor=bg_body)
    ax.add_patch(box)
    header_h = 0.05
    header = patches.Rectangle((x, y + h - header_h), w, header_h, linewidth=0, facecolor=bg_header)
    ax.add_patch(header)
    ax.text(x + w/2, y + h - 0.025, title, fontsize=10, fontweight='bold', ha='center', va='center', color='white')
    
    curr_y = y + h - header_h - 0.023
    for f in fields:
        ax.text(x + 0.02, curr_y, f, fontsize=7.8, ha='left', va='center', color='#334155', fontfamily='monospace')
        curr_y -= 0.028

paper_fields = [
    'id (PK) : String / UUID',
    'title : String',
    'authors : String',
    'publication_year : Integer',
    'abstract : Text',
    'summary : Text (100-200 Words)',
    'summary_word_count : Integer',
    'research_problem : Text',
    'methodology : Text',
    'key_findings : Array<String>',
    'limitations : Array<String>',
    'research_gaps : Array<String>',
    'conclusion : Text',
    'processing_status : Enum',
    'analysis_source : String',
    'uploaded_at : ISO-8601 UTC'
]
draw_table(0.06, 0.18, 0.44, 0.72, 'RESEARCH_PAPERS (Collection)', paper_fields, '#1e3a8a', '#eff6ff', '#3b82f6')

kw_fields = [
    'keyword : String',
    'relevance_score : Float (0-1)',
    'rank : Integer'
]
draw_table(0.60, 0.65, 0.34, 0.22, 'KEYWORDS (Embedded Array)', kw_fields, '#0f766e', '#f0fdfa', '#14b8a6')

sec_fields = [
    'section_name : String',
    'content : Text',
    'word_count : Integer'
]
draw_table(0.60, 0.37, 0.34, 0.22, 'SECTIONS (Embedded Array)', sec_fields, '#7e22ce', '#faf5ff', '#a855f7')

meta_fields = [
    'file_path : String',
    'file_size_bytes : Integer',
    'total_pages : Integer',
    'total_words : Integer'
]
draw_table(0.60, 0.09, 0.34, 0.23, 'FILE_META (Embedded Object)', meta_fields, '#334155', '#f8fafc', '#64748b')

# Connectors
ax.annotate('', xy=(0.60, 0.76), xytext=(0.50, 0.76), arrowprops=dict(arrowstyle='->', color='#0f766e', lw=1.8))
ax.text(0.55, 0.78, '1 : N', fontsize=8.5, fontweight='bold', color='#0f766e', ha='center')

ax.annotate('', xy=(0.60, 0.48), xytext=(0.50, 0.48), arrowprops=dict(arrowstyle='->', color='#7e22ce', lw=1.8))
ax.text(0.55, 0.50, '1 : N', fontsize=8.5, fontweight='bold', color='#7e22ce', ha='center')

ax.annotate('', xy=(0.60, 0.20), xytext=(0.50, 0.25), arrowprops=dict(arrowstyle='->', color='#334155', lw=1.8))
ax.text(0.55, 0.23, '1 : 1', fontsize=8.5, fontweight='bold', color='#334155', ha='center')

out2 = Path('D:/Project/implementation/research-paper-minor-project/report_figures/figure2_database_schema.png')
art2 = Path('C:/Users/Owner/.gemini/antigravity/brain/b8955fbd-9f2f-447f-969a-af43219078dd/figure2_database_schema.png')
plt.tight_layout()
plt.savefig(out2, dpi=300, facecolor='white', bbox_inches='tight')
shutil.copy(out2, art2)
print('Figure 2 regenerated successfully!')

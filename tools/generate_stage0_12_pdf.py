from __future__ import annotations

import html
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    r"C:\Users\cheny\.codex\attachments\3de0716d-5cad-4b7f-9afc-a21cf108fb5f\pasted-text.txt"
)
OUT_DIR = ROOT / "output" / "pdf"
TMP_DIR = ROOT / "tmp" / "pdfs"
PDF_PATH = OUT_DIR / "stage0_12_research_process.pdf"
FLOW_PATH = TMP_DIR / "stage0_12_flow.png"

TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_DIR / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


def choose_font() -> tuple[str, str, str]:
    candidates = [
        (r"C:\Windows\Fonts\NotoSansSC-Regular.ttf", "NotoSansSC", r"C:\Windows\Fonts\simhei.ttf"),
        (r"C:\Windows\Fonts\simhei.ttf", "SimHei", r"C:\Windows\Fonts\simhei.ttf"),
        (r"C:\Windows\Fonts\msyh.ttc", "MicrosoftYaHei", r"C:\Windows\Fonts\msyhbd.ttc"),
        (r"C:\Windows\Fonts\simsun.ttc", "SimSun", r"C:\Windows\Fonts\simhei.ttf"),
    ]
    for regular_path, regular_name, bold_path in candidates:
        if Path(regular_path).exists():
            pdfmetrics.registerFont(TTFont(regular_name, regular_path))
            bold_name = regular_name + "Bold"
            if Path(bold_path).exists():
                pdfmetrics.registerFont(TTFont(bold_name, bold_path))
            else:
                bold_name = regular_name
            return regular_name, bold_name, regular_path
    raise FileNotFoundError("No Chinese font found.")


FONT_NAME, BOLD_FONT_NAME, FONT_PATH = choose_font()


def clean_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^Today[^\n]*\n+", "", text)
    return text.strip()


def make_flowchart(font_path: str) -> None:
    from matplotlib.font_manager import FontProperties

    labels = [
        ("Stage0-2", "Understand\n（理解）"),
        ("Stage3", "Refactor\n（解耦）"),
        ("Stage4-5", "Semantic\n（二维语义）"),
        ("Stage6", "Geometry\n（三维几何）"),
        ("Stage7", "Intent\n（动作意图）"),
        ("Stage8", "Evidence\n（多源证据）"),
        ("Stage9", "Decision\n（Phys-Mem 调度）"),
        ("Stage10", "Evaluation\n（实验验证）"),
        ("Stage11", "Paper\n（论文撰写）"),
        ("Stage12", "Framework\n（未来扩展）"),
    ]
    font = FontProperties(fname=font_path)
    fig, ax = plt.subplots(figsize=(15.8, 5.2), dpi=220)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")

    palette = [
        "#E8F1FF",
        "#EAF7EF",
        "#FFF4D8",
        "#FCE8E8",
        "#F0EAFB",
        "#E7F6F5",
        "#F7ECDF",
        "#EDF0F6",
        "#F4F4EA",
        "#E9EEF0",
    ]
    edge = "#46515B"
    for idx, (stage, desc) in enumerate(labels):
        x = idx + 0.08
        box = FancyBboxPatch(
            (x, 0.55),
            0.82,
            0.84,
            boxstyle="round,pad=0.035,rounding_size=0.08",
            linewidth=1.2,
            edgecolor=edge,
            facecolor=palette[idx],
        )
        ax.add_patch(box)
        ax.text(
            x + 0.41,
            1.17,
            stage,
            ha="center",
            va="center",
            fontsize=10.5,
            fontproperties=font,
            color="#17202A",
            weight="bold",
        )
        ax.text(
            x + 0.41,
            0.82,
            desc,
            ha="center",
            va="center",
            fontsize=8.6,
            fontproperties=font,
            color="#2C3E50",
            linespacing=1.25,
        )
        if idx < len(labels) - 1:
            arrow = FancyArrowPatch(
                (x + 0.84, 0.97),
                (x + 1.0, 0.97),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.1,
                color=edge,
            )
            ax.add_patch(arrow)
    ax.text(
        5,
        1.76,
        "Stage0-Stage12 科研路线",
        ha="center",
        va="center",
        fontsize=16,
        fontproperties=font,
        color="#1F2D3D",
        weight="bold",
    )
    fig.tight_layout(pad=0.4)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FLOW_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN",
            parent=base["Title"],
            fontName=BOLD_FONT_NAME,
            fontSize=22,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17202A"),
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCN",
            parent=base["Normal"],
            fontName=FONT_NAME,
            fontSize=10.5,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#5B6670"),
            spaceAfter=18,
        ),
        "heading": ParagraphStyle(
            "HeadingCN",
            parent=base["Heading2"],
            fontName=BOLD_FONT_NAME,
            fontSize=15,
            leading=22,
            textColor=colors.HexColor("#243447"),
            spaceBefore=12,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=10.2,
            leading=16.8,
            alignment=TA_LEFT,
            firstLineIndent=0,
            textColor=colors.HexColor("#1F2933"),
            spaceAfter=8,
            wordWrap="CJK",
        ),
    }


def paragraph_for_block(block: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    escaped = html.escape(block).replace("\n", "<br/>")
    stage_heading = re.match(r"^Stage\s*\d+[^。\n]*", block, flags=re.I)
    route_heading = block.startswith("Stage0-2") or block.startswith("Stage0～Stage12")
    style = styles["heading"] if stage_heading or route_heading else styles["body"]
    return Paragraph(escaped, style)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT_NAME, 8)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawCentredString(A4[0] / 2, 12 * mm, f"{doc.page}")
    canvas.restoreState()


def build_pdf() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    make_flowchart(FONT_PATH)

    text = clean_text(SOURCE.read_text(encoding="utf-8"))
    styles = make_styles()
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Stage0-Stage12 Research Process",
        author="Codex",
    )

    story = [
        Paragraph("Stage0-Stage12 完整科研流程整理", styles["title"]),
        Paragraph("以下内容按原文整理，正文不改写。", styles["subtitle"]),
        Image(str(FLOW_PATH), width=170 * mm, height=56 * mm),
        Spacer(1, 7 * mm),
        PageBreak(),
    ]
    blocks = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    for block in blocks:
        story.append(paragraph_for_block(block, styles))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build_pdf()
    print(PDF_PATH)

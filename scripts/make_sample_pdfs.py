"""One-off: render data/*.md into data/*.pdf (real PDF text extraction target),
plus a generated architecture diagram embedded as a raster image in
product_overview.pdf (real embedded-image target for vision captioning).
Run once, then the .md sources are removed so data/ holds only PDFs.

python scripts/make_sample_pdfs.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer

DATA_DIR = Path(__file__).parent.parent / "data"
DIAGRAM_PATH = DATA_DIR / "_sync_architecture.png"


def make_architecture_diagram():
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.axis("off")

    boxes = {
        "client": (0.05, 0.4, "Desktop Sync\nClient"),
        "edge": (0.38, 0.4, "Nimbus Edge API\n(Sync Engine v3)"),
        "us": (0.7, 0.65, "Primary:\nus-east-1"),
        "eu": (0.7, 0.4, "Replica:\neu-west-1"),
        "ap": (0.7, 0.15, "Replica:\nap-southeast-1"),
    }
    for x, y, label in boxes.values():
        ax.add_patch(plt.Rectangle((x, y), 0.24, 0.18, fill=False, edgecolor="black"))
        ax.text(x + 0.12, y + 0.09, label, ha="center", va="center", fontsize=8)

    ax.annotate("", xy=(0.38, 0.49), xytext=(0.29, 0.49), arrowprops=dict(arrowstyle="->"))
    for target_y in (0.74, 0.49, 0.24):
        ax.annotate("", xy=(0.7, target_y), xytext=(0.62, 0.49), arrowprops=dict(arrowstyle="->"))

    ax.text(0.5, 0.02, "~5s propagation to all replica regions", ha="center", fontsize=8, style="italic")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(DIAGRAM_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def md_to_pdf(md_path, pdf_path, embed_diagram=False):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(pdf_path), pagesize=LETTER)
    flow = []

    for line in md_path.read_text().splitlines():
        if not line.strip():
            flow.append(Spacer(1, 8))
        elif line.startswith("# "):
            flow.append(Paragraph(line[2:], styles["Title"]))
        elif line.startswith("## "):
            flow.append(Paragraph(line[3:], styles["Heading2"]))
        else:
            flow.append(Paragraph(line, styles["BodyText"]))

    if embed_diagram:
        flow.append(Spacer(1, 12))
        flow.append(Paragraph("Sync Architecture", styles["Heading2"]))
        flow.append(Image(str(DIAGRAM_PATH), width=380, height=200))

    doc.build(flow)


def main():
    make_architecture_diagram()
    md_paths = sorted(DATA_DIR.glob("*.md"))
    for md_path in md_paths:
        pdf_path = md_path.with_suffix(".pdf")
        md_to_pdf(md_path, pdf_path, embed_diagram=(md_path.stem == "product_overview"))
        print(f"wrote {pdf_path.name}")

    DIAGRAM_PATH.unlink()
    for md_path in md_paths:
        md_path.unlink()
    print(f"removed {len(md_paths)} .md sources — data/ now holds PDFs only")


if __name__ == "__main__":
    main()

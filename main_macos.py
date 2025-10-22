#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import tempfile


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run(cmd, cwd=None):
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc




def _sanitize_filename(name: str) -> str:
    name = name.strip()
    # Replace spaces with underscores and remove forbidden characters
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^\w\-\.\u4e00-\u9fff]+", "", name)  # keep CJK
    # Limit length
    return name[:150] or "figure"


def _convert_to_png(src: Path, dest_png: Path):
    """Convert/copy an image file to PNG at `dest_png` without relying on `sips`.

    - If already PNG, copy as-is.
    - Else try Pillow if available; if not, copy with original extension next to desired path.
    Returns the actual written path.
    """
    dest_png.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".png":
        shutil.copy2(src, dest_png)
        return dest_png
    try:
        from PIL import Image  # type: ignore

        with Image.open(src) as im:
            im.save(dest_png, format="PNG")
        return dest_png
    except Exception:
        alt = dest_png.with_suffix(src.suffix.lower())
        shutil.copy2(src, alt)
        return alt


def _attempt_repair_pdf(src_pdf: Path) -> Path | None:
    """Try to repair a problematic PDF using qpdf or Ghostscript if available.

    Returns a path to a repaired temporary PDF if successful; otherwise None.
    """
    repaired = None
    tmpdir = Path(tempfile.mkdtemp(prefix=f"repair_{src_pdf.stem}_"))
    # Try qpdf
    if have("qpdf"):
        out_pdf = tmpdir / "repaired.pdf"
        try:
            run(["qpdf", str(src_pdf), str(out_pdf)])
            if out_pdf.exists() and out_pdf.stat().st_size > 0:
                repaired = out_pdf
                return repaired
        except Exception:
            pass
    # Try Ghostscript
    if have("gs"):
        out_pdf = tmpdir / "repaired.pdf"
        try:
            run([
                "gs", "-o", str(out_pdf), "-sDEVICE=pdfwrite",
                "-dCompatibilityLevel=1.5", "-dNOPAUSE", "-dBATCH", str(src_pdf)
            ])
            if out_pdf.exists() and out_pdf.stat().st_size > 0:
                repaired = out_pdf
                return repaired
        except Exception:
            pass
    return None


def _render_pages_to_png(pdf_path: Path, out_dir: Path, dpi: int = 150) -> int:
    """Fallback: render each page to a PNG with pdftoppm and name as page{n}_img1.png."""
    if not have("pdftoppm"):
        return 0
    with tempfile.TemporaryDirectory(prefix=f"raster_{pdf_path.stem}_") as tmpd:
        tmp = Path(tmpd)
        prefix = tmp / "page"
        run(["pdftoppm", "-png", "-r", str(dpi), str(pdf_path.resolve()), str(prefix)])
        pages = sorted(tmp.glob("page-*.png"))
        saved = 0
        for p in pages:
            # page-1.png -> 1
            m = re.search(r"page-(\d+)\.png$", p.name)
            idx = m.group(1) if m else str(saved + 1)
            dest = out_dir / f"page{idx}_img1.png"
            shutil.copy2(p, dest)
            saved += 1
        return saved


def extract_figures_with_titles(pdf_path: Path, out_dir: Path, proximity: int = 80) -> int:
    """Extract per-page images and name them by their nearest 'Figure' caption.

    Uses pdftohtml -xml to get positioned text and per-image raster assets,
    then associates images to captions like 'Fig.R01.1 ...' or 'Figure ...' or '图...'.
    """
    if not have("pdftohtml"):
        raise RuntimeError("Required tool 'pdftohtml' not found in PATH. Please install poppler (pdftohtml).")

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    used_names = set()

    # Use a temporary working directory for XML and extracted assets
    with tempfile.TemporaryDirectory(prefix=f"{pdf_path.stem}_") as tmpd:
        xml_dir = Path(tmpd)
        xml_base = Path("doc")  # run in xml_dir so outputs land there
        xml_file = xml_dir / "doc.xml"

        # Produce XML + extracted images
        try:
            run(["pdftohtml", "-xml", "-hidden", str(pdf_path.resolve()), str(xml_base)], cwd=xml_dir)
        except Exception:
            # Try repair once
            repaired = _attempt_repair_pdf(pdf_path)
            if repaired is not None:
                run(["pdftohtml", "-xml", "-hidden", str(repaired.resolve()), str(xml_base)], cwd=xml_dir)
            else:
                # Fallback to rasterizing pages; captions unavailable
                return _render_pages_to_png(pdf_path, out_dir)

        if not xml_file.exists():
            # poppler names as <base>.xml when provided base path
            candidates = list(xml_dir.glob("*.xml"))
            if not candidates:
                # As a last resort, render pages
                return _render_pages_to_png(pdf_path, out_dir)
            xml_file = candidates[0]

        # Parse XML
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except Exception:
            # If XML parsing fails, fallback to raster pages
            return _render_pages_to_png(pdf_path, out_dir)

        # Regex for caption detection
        caption_re = re.compile(r"^(?:Fig(?:ure)?\.?|图)\s*[:：\.]?\s*.*", re.IGNORECASE)

        for page in root.findall("page"):
            pnum = int(page.get("number", "0"))

            # Collect text segments on the page
            segments = []
            for tx in page.findall("text"):
                raw_text = "".join(tx.itertext())
                if not raw_text:
                    continue
                trimmed = raw_text.strip()
                if not trimmed:
                    continue
                segments.append({
                    "top": float(tx.get("top", 0.0)),
                    "left": float(tx.get("left", 0.0)),
                    "width": float(tx.get("width", 0.0)),
                    "height": float(tx.get("height", 0.0)),
                    "text": trimmed,
                })

            # Group segments into lines by vertical proximity
            line_tol = 2.0
            lines = []
            for seg in sorted(segments, key=lambda s: s["top"]):
                placed = False
                for line in lines:
                    if abs(seg["top"] - line["top"]) <= line_tol:
                        line["segments"].append(seg)
                        line["top"] = min(line["top"], seg["top"])
                        line["bottom"] = max(line["bottom"], seg["top"] + seg["height"])
                        placed = True
                        break
                if not placed:
                    lines.append({
                        "top": seg["top"],
                        "bottom": seg["top"] + seg["height"],
                        "segments": [seg],
                    })

            # Collect caption texts from grouped lines
            caps = []
            for line in lines:
                segs = sorted(line["segments"], key=lambda s: s["left"])
                if not segs:
                    continue

                # Further split line segments into chunks separated by large horizontal gaps (e.g., multi-column captions)
                chunk_gap = 40.0
                chunks = []
                current = [segs[0]]
                for seg in segs[1:]:
                    prev = current[-1]
                    gap = seg["left"] - (prev["left"] + prev["width"])
                    if gap > chunk_gap:
                        chunks.append(current)
                        current = [seg]
                    else:
                        current.append(seg)
                if current:
                    chunks.append(current)

                for chunk in chunks:
                    # Split within chunk when a new caption-like segment starts
                    group_segments = []
                    active = []
                    for s in chunk:
                        txt = (s.get("text") or "").strip()
                        if not txt:
                            continue
                        if active:
                            if caption_re.match(txt):
                                group_segments.append(active)
                                active = [s]
                            else:
                                active.append(s)
                        else:
                            if caption_re.match(txt):
                                active = [s]
                    if active:
                        group_segments.append(active)

                    for segs2 in group_segments:
                        parts = [s.get("text", "").strip() for s in segs2 if s.get("text")]
                        if not parts:
                            continue
                        text = " ".join(parts)
                        left = min(s["left"] for s in segs2)
                        right = max(s["left"] + s["width"] for s in segs2)
                        top = min(s["top"] for s in segs2)
                        bottom = max(s["top"] + s["height"] for s in segs2)
                        caps.append({
                            "rect": (left, top, right, bottom),
                            "text": text,
                        })

            # Collect images
            imgs = []
            for im in page.findall("image"):
                top = float(im.get("top", 0))
                left = float(im.get("left", 0))
                width = float(im.get("width", 0))
                height = float(im.get("height", 0))
                src = im.get("src")
                if src:
                    imgs.append({
                        "rect": (left, top, left + width, top + height),
                        "src": Path(src),
                    })

            if not imgs:
                continue

            # Match each image to the nearest caption
            for idx, im in enumerate(imgs, start=1):
                l, t, r, b = im["rect"]
                icx = (l + r) / 2
                # Prefer captions just below the image within proximity
                best = None
                best_score = 1e9
                for cap in caps:
                    cl, ct, cr, cb = cap["rect"]
                    ccx = (cl + cr) / 2
                    vertical_gap = ct - b  # positive if caption below image
                    above_gap = t - cb      # positive if caption above image
                    horiz_dist = abs(ccx - icx)
                    score = None
                    if 0 <= vertical_gap <= proximity:
                        score = vertical_gap + horiz_dist * 0.05
                    elif 0 <= above_gap <= proximity:
                        score = above_gap + horiz_dist * 0.1 + 100  # slightly worse than below
                    else:
                        # Fallback to center distance if within a loose rectangle
                        if (abs(ct - b) <= proximity * 1.5) or (abs(t - cb) <= proximity * 1.5):
                            score = min(abs(ct - b), abs(t - cb)) + horiz_dist * 0.2 + 200
                    if score is not None and score < best_score:
                        best_score = score
                        best = cap

                if best is None:
                    title = f"page{pnum}_img{idx}"
                else:
                    raw = best["text"].strip()
                    # Remove trailing/leading punctuation and inner spaces normalization
                    title = _sanitize_filename(raw)
                    # Ensure uniqueness
                    base_title = title
                    n = 1
                    while title in used_names:
                        n += 1
                        title = f"{base_title}_{n}"
                    used_names.add(title)

                src_attr = im["src"]
                src_path = Path(src_attr)
                if not src_path.is_absolute():
                    src_path = (xml_file.parent / src_path)
                dest_png = out_dir / f"{title}.png"
                _convert_to_png(src_path, dest_png)
                saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser(description="Batch extract figure images from PDFs and name them by their Figure Titles.")
    parser.add_argument("input", type=Path, help="Input directory (or single PDF file)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output directory. Creates a subfolder per PDF (by file name).")
    parser.add_argument("--proximity", type=int, default=80, help="Max vertical distance (px) to match a caption to an image")
    parser.add_argument("--recursive", action="store_true", help="Recursively find PDFs under input directory")

    args = parser.parse_args()

    inp = args.input
    out_root = args.output

    if not inp.exists():
        print(f"Input path not found: {inp}", file=sys.stderr)
        sys.exit(1)

    # Gather PDFs
    if inp.is_dir():
        pdfs = list(inp.rglob("*.pdf")) if args.recursive else list(inp.glob("*.pdf"))
    elif inp.suffix.lower() == ".pdf":
        pdfs = [inp]
    else:
        print("Input must be a PDF file or a directory containing PDFs.", file=sys.stderr)
        sys.exit(1)

    if not pdfs:
        print("No PDF files found to process.")
        return

    out_root.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    for pdf in pdfs:
        sub_out = out_root / pdf.stem
        sub_out.mkdir(parents=True, exist_ok=True)
        try:
            saved = extract_figures_with_titles(pdf, sub_out, proximity=args.proximity)
            total_saved += saved
            print(f"Processed {pdf.name}: saved {saved} image(s) -> {sub_out}")
        except Exception as e:
            print(f"Failed {pdf}: {e}", file=sys.stderr)

    print(f"Done. Total images saved: {total_saved}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

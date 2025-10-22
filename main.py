#!/usr/bin/env python3
import argparse
import os
import re
import sys
import threading
import time
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "PyMuPDF (package name `pymupdf`) is required. Install it with `pip install pymupdf`."
    ) from exc


CAPTION_PREFIXES = ("fig", "figure", "\u56fe", "\u5716")
Logger = Optional[Callable[[str], None]]


def _emit(logger: Logger, message: str, *, error: bool = False) -> None:
    if logger:
        logger(message)
    else:
        stream = sys.stderr if error else sys.stdout
        print(message, file=stream)


def _sanitize_filename(name: str) -> str:
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"[^\w\-\.\u4e00-\u9fff]+", "", name)
    return name[:150] or "figure"


def have(cmd: str) -> bool:
    """Return True if an executable is available in PATH.

    Works cross-platform (e.g. finds `pdftohtml.exe` on Windows).
    """
    return shutil.which(cmd) is not None


def _inject_embedded_bin_into_path() -> None:
    """When running as a PyInstaller onefile, prepend the bundled `bin` to PATH.

    This allows embedded Poppler executables (pdftohtml/pdftoppm) to be found
    without requiring external installation.
    """
    try:
        base = getattr(sys, "_MEIPASS", None)
        if not base:
            return
        bin_dir = Path(base) / "bin"
        if bin_dir.exists():
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        # Non-fatal; simply skip PATH injection
        pass


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def _convert_to_png(src: Path, dest_png: Path) -> Path:
    """Convert/copy an image file to PNG at `dest_png`.

    - If already PNG, copy as-is.
    - Else try Pillow if available; if not, copy with original extension next to desired path.
    Returns the actual written path.
    """
    dest_png.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".png":
        shutil.copy2(src, dest_png)
        return dest_png
    # Try Pillow-based conversion
    try:
        from PIL import Image  # type: ignore

        with Image.open(src) as im:
            im.save(dest_png, format="PNG")
        return dest_png
    except Exception:
        # Fallback: keep original format to avoid mismatched extensions
        alt = dest_png.with_suffix(src.suffix.lower())
        shutil.copy2(src, alt)
        return alt


def _attempt_repair_pdf(src_pdf: Path) -> Path | None:
    """Try to repair a problematic PDF using qpdf or Ghostscript if available.

    Returns a path to a repaired temporary PDF if successful; otherwise None.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix=f"repair_{src_pdf.stem}_"))
    # Try qpdf
    if have("qpdf"):
        out_pdf = tmpdir / "repaired.pdf"
        try:
            run(["qpdf", str(src_pdf), str(out_pdf)])
            if out_pdf.exists() and out_pdf.stat().st_size > 0:
                return out_pdf
        except Exception:
            pass
    # Try Ghostscript
    if have("gs"):
        out_pdf = tmpdir / "repaired.pdf"
        try:
            run([
                "gs",
                "-o",
                str(out_pdf),
                "-sDEVICE=pdfwrite",
                "-dCompatibilityLevel=1.5",
                "-dNOPAUSE",
                "-dBATCH",
                str(src_pdf),
            ])
            if out_pdf.exists() and out_pdf.stat().st_size > 0:
                return out_pdf
        except Exception:
            pass
    return None


def _render_pages_to_png(pdf_path: Path, out_dir: Path, dpi: int = 150, *, log: Logger = None) -> int:
    """Fallback: render each page to a PNG with pdftoppm.

    Note: pageX_imgY assets are not needed. We only log them as skipped
    instead of writing files to disk.
    """
    if not have("pdftoppm"):
        return 0
    with tempfile.TemporaryDirectory(prefix=f"raster_{pdf_path.stem}_") as tmpd:
        tmp = Path(tmpd)
        prefix = tmp / "page"
        run(["pdftoppm", "-png", "-r", str(dpi), str(pdf_path.resolve()), str(prefix)])
        pages = sorted(tmp.glob("page-*.png"))
        saved = 0
        for p in pages:
            m = re.search(r"page-(\d+)\.png$", p.name)
            idx = m.group(1) if m else str(saved + 1)
            _emit(
                log,
                f"[SKIP] {pdf_path.name} page {idx} image 1: fallback rasterized page (pageX_imgY) not saved",
            )
            # Intentionally do not save fallback images
        return saved


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _line_text(line: dict) -> str:
    spans = line.get("spans", [])
    text_parts = []
    for span in spans:
        span_text = span.get("text")
        if not span_text:
            continue
        cleaned = span_text.strip()
        if cleaned:
            text_parts.append(cleaned)
    return _normalize_text(" ".join(text_parts))


def _looks_like_caption(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    lower = normalized.lower()
    return any(lower.startswith(prefix) for prefix in CAPTION_PREFIXES)


def _collect_captions(blocks: list[dict]) -> list[dict]:
    captions: list[dict] = []

    def make_group(seg_list: list[dict]) -> dict:
        segs = list(seg_list)
        left = min(seg.get("left", 0.0) for seg in segs)
        right = max(seg.get("left", 0.0) + seg.get("width", 0.0) for seg in segs)
        top = min(seg.get("top", 0.0) for seg in segs)
        bottom = max(seg.get("top", 0.0) + seg.get("height", 0.0) for seg in segs)
        return {
            "segments": segs,
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
            "center": (left + right) / 2,
        }

    def extend_group(group: dict, seg: dict) -> None:
        group["segments"].append(seg)
        left = seg.get("left", 0.0)
        width = seg.get("width", 0.0)
        top = seg.get("top", 0.0)
        height = seg.get("height", 0.0)
        group["left"] = min(group["left"], left)
        group["right"] = max(group["right"], left + width)
        group["top"] = min(group["top"], top)
        group["bottom"] = max(group["bottom"], top + height)
        group["center"] = (group["left"] + group["right"]) / 2

    for block in blocks:
        if block.get("type") != 0:  # only text blocks
            continue
        lines = block.get("lines", [])
        i = 0
        while i < len(lines):
            line = lines[i]
            segs = sorted(line.get("segments", []), key=lambda s: s.get("left", 0.0))
            if not segs:
                i += 1
                continue

            chunk_gap = 40.0
            chunks: list[list[dict]] = []
            current: list[dict] = []
            for seg in segs:
                if current:
                    prev = current[-1]
                    gap = seg.get("left", 0.0) - (prev.get("left", 0.0) + prev.get("width", 0.0))
                    if gap > chunk_gap:
                        chunks.append(current)
                        current = [seg]
                    else:
                        current.append(seg)
                else:
                    current = [seg]
            if current:
                chunks.append(current)

            groups: list[dict] = []
            for chunk in chunks:
                active: list[dict] = []
                for seg in chunk:
                    seg_text = (seg.get("text") or "").strip()
                    if not seg_text:
                        continue
                    if active:
                        if _looks_like_caption(seg_text):
                            groups.append(make_group(active))
                            active = [seg]
                        else:
                            active.append(seg)
                    else:
                        if _looks_like_caption(seg_text):
                            active = [seg]
                if active:
                    groups.append(make_group(active))

            if not groups:
                i += 1
                continue

            next_idx = i + 1
            merged_lines = 0
            stop_merging = False

            while next_idx < len(lines) and merged_lines < 2:
                next_line = lines[next_idx]
                next_segs = sorted(next_line.get("segments", []), key=lambda s: s.get("left", 0.0))
                if not next_segs:
                    next_idx += 1
                    merged_lines += 1
                    continue

                next_top = min(seg.get("top", 0.0) for seg in next_segs)
                current_bottom = max(group["bottom"] for group in groups)
                if next_top - current_bottom > 20:
                    stop_merging = True
                    break

                if any(_looks_like_caption((seg.get("text") or "").strip()) for seg in next_segs):
                    stop_merging = True
                    break

                for seg in next_segs:
                    seg_text = (seg.get("text") or "").strip()
                    if not seg_text:
                        continue
                    target = min(
                        groups,
                        key=lambda group: abs(
                            (seg.get("left", 0.0) + seg.get("width", 0.0) / 2) - group["center"]
                        ),
                    )
                    extend_group(target, seg)

                next_idx += 1
                merged_lines += 1

            for group in groups:
                text = _normalize_text(
                    " ".join(seg.get("text", "").strip() for seg in group["segments"] if seg.get("text"))
                )
                if text and _looks_like_caption(text):
                    captions.append(
                        {
                            "rect": (group["left"], group["top"], group["right"], group["bottom"]),
                            "text": text,
                        }
                    )

            if stop_merging:
                i += 1
            else:
                i = next_idx
    return captions


def _collect_images(page: fitz.Page, blocks: list[dict]) -> list[dict]:
    images: list[dict] = []
    seen = set()

    for block in blocks:
        if block.get("type") != 1:
            continue
        bbox = block.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        image_info = block.get("image")
        stream: bytes | None = None
        xref: int | None = None

        if isinstance(image_info, dict):
            xref = image_info.get("xref")
            raw_stream = image_info.get("image")
            if isinstance(raw_stream, (bytes, bytearray)):
                stream = bytes(raw_stream)
        elif isinstance(image_info, (bytes, bytearray)):
            stream = bytes(image_info)
        elif isinstance(image_info, (int, float)):
            xref = int(image_info)
        elif isinstance(image_info, str):
            try:
                xref = int(image_info)
            except ValueError:
                stream = image_info.encode("latin1")

        if xref is None:
            fallback = block.get("xref")
            if isinstance(fallback, str):
                try:
                    fallback = int(fallback)
                except ValueError:
                    fallback = None
            if isinstance(fallback, (int, float)):
                xref = int(fallback)

        if xref is not None:
            stream = None

        if xref is None and stream is None:
            continue

        key = (tuple(bbox), xref, stream[:16] if stream else None)
        if key in seen:
            continue
        seen.add(key)

        images.append(
            {
                "rect": tuple(bbox),
                "xref": xref,
                "stream": stream,
            }
        )

    block_xrefs = {img["xref"] for img in images if img.get("xref") is not None}
    for info in page.get_images(full=True):
        xref = info[0]
        if xref in block_xrefs:
            continue
        images.append(
            {
                "rect": None,
                "xref": xref,
                "stream": None,
            }
        )

    return images


def _pick_caption_for_image(image: dict, captions: list[dict], proximity: int) -> dict | None:
    if not captions or image.get("rect") is None:
        return None
    l, t, r, b = image["rect"]
    icx = (l + r) / 2
    best = None
    best_score = float("inf")
    for caption in captions:
        cl, ct, cr, cb = caption["rect"]
        ccx = (cl + cr) / 2
        vertical_gap = ct - b
        above_gap = t - cb
        horiz_dist = abs(ccx - icx)
        score = None
        if 0 <= vertical_gap <= proximity:
            score = vertical_gap + horiz_dist * 0.05
        elif 0 <= above_gap <= proximity:
            score = above_gap + horiz_dist * 0.1 + 100
        else:
            vertical_distance = min(abs(ct - b), abs(t - cb))
            if vertical_distance <= proximity * 1.5:
                score = vertical_distance + horiz_dist * 0.2 + 200
        if score is not None and score < best_score:
            best_score = score
            best = caption
    return best


def _save_pixmap(doc: fitz.Document, image: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    xref = image.get("xref")
    stream = image.get("stream")

    if xref is not None:
        stream = None

    if stream:
        dest.write_bytes(stream)
        return

    if xref is None:
        raise RuntimeError("Image lacks both xref and stream data.")

    pix = fitz.Pixmap(doc, xref)
    pix_to_save = pix
    try:
        if pix.n - pix.alpha > 3:  # convert CMYK and similar to RGB
            pix_to_save = fitz.Pixmap(fitz.csRGB, pix)
        pix_to_save.save(dest)
    finally:
        if pix_to_save is not pix:
            pix_to_save = None
        pix = None


def _extract_figures_with_titles_mupdf(
    pdf_path: Path,
    out_dir: Path,
    proximity: int = 80,
    log: Logger = None,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    used_names: set[str] = set()

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # pragma: no cover - surface open errors nicely
        raise RuntimeError(f"Failed to open {pdf_path}: {exc}") from exc

    with doc:
        for page_index, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict")
            blocks = page_dict.get("blocks", [])
            captions = _collect_captions(blocks)
            images = _collect_images(page, blocks)
            if not images:
                continue

            for idx, image in enumerate(images, start=1):
                caption = _pick_caption_for_image(image, captions, proximity)
                if caption is None:
                    _emit(
                        log,
                        f"[SKIP] {pdf_path.name} page {page_index} image {idx}: no caption within proximity={proximity}",
                    )
                    continue

                raw_caption = _normalize_text(caption["text"])
                base_title = _sanitize_filename(raw_caption)
                title = base_title or f"caption_{page_index}_{idx}"
                counter = 1
                while title in used_names:
                    counter += 1
                    title = f"{base_title}_{counter}"
                used_names.add(title)

                dest_png = out_dir / f"{title}.png"
                _save_pixmap(doc, image, dest_png)
                saved += 1
                _emit(
                    log,
                    f"[SAVED] {pdf_path.name} page {page_index} image {idx} -> {dest_png.name} (caption: {raw_caption})",
                )

    return saved


def extract_figures_with_titles_poppler(
    pdf_path: Path,
    out_dir: Path,
    proximity: int = 80,
    *,
    log: Logger = None,
) -> int:
    """Extract per-page images and name them by the nearest caption using Poppler's pdftohtml XML.

    This mirrors the logic in main_macos.py to improve results for PDFs where PyMuPDF yields no images/positions.
    """
    if not have("pdftohtml"):
        raise RuntimeError("Required tool 'pdftohtml' not found in PATH. Please install Poppler (pdftohtml).")

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    used_names: set[str] = set()

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
                return _render_pages_to_png(pdf_path, out_dir, log=log)

        if not xml_file.exists():
            # poppler names as <base>.xml when provided base path
            candidates = list(xml_dir.glob("*.xml"))
            if not candidates:
                # As a last resort, render pages
                return _render_pages_to_png(pdf_path, out_dir, log=log)
            xml_file = candidates[0]

        # Parse XML
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except Exception:
            # If XML parsing fails, fallback to raster pages
            return _render_pages_to_png(pdf_path, out_dir, log=log)

        # Regex for caption detection (Fig/Figure/图/圖 etc.)
        caption_re = re.compile(r"^(?:fig(?:ure)?\.?\s*|[图圖]\s*[:：\.、]?\s*)", re.IGNORECASE)

        for page in root.findall("page"):
            pnum_text = page.get("number", "0")
            try:
                pnum = int(pnum_text)
            except ValueError:
                pnum = 0

            # Collect text segments on the page
            segments: list[dict] = []
            for tx in page.findall("text"):
                raw_text = "".join(tx.itertext())
                if not raw_text:
                    continue
                trimmed = raw_text.strip()
                if not trimmed:
                    continue
                try:
                    top = float(tx.get("top", 0.0))
                    left = float(tx.get("left", 0.0))
                    width = float(tx.get("width", 0.0))
                    height = float(tx.get("height", 0.0))
                except Exception:
                    continue
                segments.append({
                    "top": top,
                    "left": left,
                    "width": width,
                    "height": height,
                    "text": trimmed,
                })

            # Group segments into lines by vertical proximity
            line_tol = 2.0
            lines: list[dict] = []
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
            caps: list[dict] = []
            for line in lines:
                segs = sorted(line["segments"], key=lambda s: s["left"])
                if not segs:
                    continue

                # Further split line segments into chunks separated by large horizontal gaps (e.g., multi-column captions)
                chunk_gap = 40.0
                chunks: list[list[dict]] = []
                current: list[dict] = [segs[0]]
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
                    # Within a chunk, split into caption groups whenever a new caption-like segment starts
                    group_segments: list[list[dict]] = []
                    active: list[dict] = []
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

                    for segs in group_segments:
                        parts = [s.get("text", "").strip() for s in segs if s.get("text")]
                        if not parts:
                            continue
                        text = " ".join(parts)
                        left = min(s["left"] for s in segs)
                        right = max(s["left"] + s["width"] for s in segs)
                        top = min(s["top"] for s in segs)
                        bottom = max(s["top"] + s["height"] for s in segs)
                        caps.append({
                            "rect": (left, top, right, bottom),
                            "text": text,
                        })

            # Collect images on page
            imgs: list[dict] = []
            for im in page.findall("image"):
                try:
                    top = float(im.get("top", 0))
                    left = float(im.get("left", 0))
                    width = float(im.get("width", 0))
                    height = float(im.get("height", 0))
                except Exception:
                    continue
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
                    _emit(
                        log,
                        f"[SKIP] {pdf_path.name} page {pnum} image {idx}: no caption within proximity={proximity}",
                    )
                    continue
                else:
                    raw = best["text"].strip()
                    title = _sanitize_filename(raw)
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
                _emit(
                    log,
                    f"[SAVED] {pdf_path.name} page {pnum} image {idx} -> {dest_png.name} (caption: {raw})",
                )

    return saved


def extract_figures_with_titles(
    pdf_path: Path,
    out_dir: Path,
    proximity: int = 80,
    log: Logger = None,
) -> int:
    """Wrapper that tries Poppler-based extraction first, then falls back to PyMuPDF.

    This change aligns Windows behavior with main_macos.py to avoid empty results.
    """
    # Try Poppler pipeline first if available
    if have("pdftohtml"):
        try:
            saved = extract_figures_with_titles_poppler(pdf_path, out_dir, proximity, log=log)
            if saved > 0:
                return saved
        except Exception as exc:
            _emit(log, f"[WARN] Poppler pipeline failed for {pdf_path.name}: {exc}. Falling back to PyMuPDF.")

    # Fallback to original PyMuPDF-based pipeline
    return _extract_figures_with_titles_mupdf(pdf_path, out_dir, proximity, log)


def process_pdfs(
    input_path: Path,
    output_root: Path,
    *,
    proximity: int = 80,
    recursive: bool = False,
    log: Logger = None,
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_dir():
        pdfs = list(input_path.rglob("*.pdf")) if recursive else list(input_path.glob("*.pdf"))
    elif input_path.suffix.lower() == ".pdf":
        pdfs = [input_path]
    else:
        raise ValueError("Input must be a PDF file or a directory containing PDFs.")

    if not pdfs:
        _emit(log, "No PDF files found to process.")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    start_time = time.perf_counter()

    for pdf in pdfs:
        pdf_start = time.perf_counter()
        sub_out = output_root / pdf.stem
        sub_out.mkdir(parents=True, exist_ok=True)
        try:
            saved = extract_figures_with_titles(pdf, sub_out, proximity=proximity, log=log)
            total_saved += saved
            elapsed = time.perf_counter() - pdf_start
            _emit(log, f"Processed {pdf.name}: saved {saved} image(s) -> {sub_out} (time: {elapsed:.2f}s)")
        except Exception as exc:
            elapsed = time.perf_counter() - pdf_start
            _emit(log, f"Failed {pdf} after {elapsed:.2f}s: {exc}", error=True)

    total_elapsed = time.perf_counter() - start_time
    pdf_count = len(pdfs)
    average = total_elapsed / pdf_count if pdf_count else 0.0

    _emit(log, f"Done. Total images saved: {total_saved}")
    _emit(log, f"Total processing time: {total_elapsed:.2f}s (average {average:.2f}s per PDF)")
    return total_saved


def run_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText

    root = tk.Tk()
    root.title("ExtractPDF")

    input_var = tk.StringVar()
    output_var = tk.StringVar()
    proximity_var = tk.StringVar(value="80")

    mainframe = ttk.Frame(root, padding=12)
    mainframe.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    def _select_directory(target_var: tk.StringVar, title: str) -> None:
        path = filedialog.askdirectory(title=title)
        if path:
            target_var.set(path)

    def _append_log(message: str) -> None:
        log_box.configure(state="normal")
        log_box.insert("end", message + "\n")
        log_box.see("end")
        log_box.configure(state="disabled")

    def append_log(message: str) -> None:
        root.after(0, _append_log, message)
        # Mirror logs to stdout to ease debugging when launched from terminal
        print(message)

    def run_extraction() -> None:
        inp = input_var.get().strip()
        out = output_var.get().strip()
        proximity_text = proximity_var.get().strip() or "80"

        if not inp:
            messagebox.showerror("Missing input", "Please choose an input directory or PDF file.")
            return
        if not out:
            messagebox.showerror("Missing output", "Please choose an output directory.")
            return
        try:
            proximity_value = int(proximity_text)
        except ValueError:
            messagebox.showerror("Invalid proximity", "Proximity must be an integer value.")
            return

        start_button.config(state="disabled")
        log_box.configure(state="normal")
        log_box.delete("1.0", "end")
        log_box.configure(state="disabled")

        def worker() -> None:
            try:
                append_log("Starting extraction...")
                process_pdfs(
                    Path(inp),
                    Path(out),
                    proximity=proximity_value,
                    recursive=False,
                    log=append_log,
                )
            except Exception as exc:  # surface error dialog from main thread
                append_log(f"[ERROR] {exc}")
                root.after(0, lambda: messagebox.showerror("Extraction failed", str(exc)))
            finally:
                root.after(0, lambda: start_button.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    ttk.Label(mainframe, text="Input path:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
    input_entry = ttk.Entry(mainframe, textvariable=input_var, width=50)
    input_entry.grid(row=0, column=1, sticky="ew", pady=(0, 6))
    ttk.Button(mainframe, text="Browse…", command=lambda: _select_directory(input_var, "Select input path")).grid(
        row=0,
        column=2,
        padx=(8, 0),
        pady=(0, 6),
    )

    ttk.Label(mainframe, text="Output directory:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
    output_entry = ttk.Entry(mainframe, textvariable=output_var, width=50)
    output_entry.grid(row=1, column=1, sticky="ew", pady=(0, 6))
    ttk.Button(mainframe, text="Browse…", command=lambda: _select_directory(output_var, "Select output directory")).grid(
        row=1,
        column=2,
        padx=(8, 0),
        pady=(0, 6),
    )

    ttk.Label(mainframe, text="Proximity (points):").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(0, 10))
    proximity_entry = ttk.Entry(mainframe, textvariable=proximity_var, width=12)
    proximity_entry.grid(row=2, column=1, sticky="w", pady=(0, 10))

    start_button = ttk.Button(mainframe, text="Start extraction", command=run_extraction)
    start_button.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 12))

    log_box = ScrolledText(mainframe, height=18, state="disabled")
    log_box.grid(row=4, column=0, columnspan=3, sticky="nsew")

    mainframe.columnconfigure(1, weight=1)
    mainframe.rowconfigure(4, weight=1)

    root.mainloop()


def main() -> None:
    # Ensure embedded tools are discoverable if running from a onefile bundle
    _inject_embedded_bin_into_path()

    parser = argparse.ArgumentParser(
        description="Batch extract figure images from PDFs using PyMuPDF and name them by their Figure titles.",
    )
    parser.add_argument("input", type=Path, nargs="?", help="Input directory (or single PDF file)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory. Creates a subfolder per PDF (by file name).",
    )
    parser.add_argument(
        "--proximity",
        type=int,
        default=80,
        help="Maximum vertical distance (in PDF points) between an image and its caption.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively find PDFs under input directory",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the graphical interface instead of the command line workflow.",
    )

    args = parser.parse_args()

    # If packaged as an executable (PyInstaller) with no CLI args, launch GUI by default
    is_frozen = getattr(sys, "frozen", False)
    no_cli_args = len(sys.argv) == 1
    if args.gui or (is_frozen and no_cli_args):
        run_gui()
        return

    if args.input is None or args.output is None:
        parser.error("input and --output are required unless --gui is used")

    try:
        process_pdfs(
            args.input,
            args.output,
            proximity=args.proximity,
            recursive=args.recursive,
            log=None,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - surface entry point issues
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

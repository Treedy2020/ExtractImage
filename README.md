ExtractPDF

Overview
- Extracts figure images from PDFs and names them using nearby captions: prefixes like `Fig`, `Figure`, `图/圖` are supported.
- Prefers a Poppler (pdftohtml XML) pipeline for robust caption/image positions, falls back to PyMuPDF when needed.
- Skips generic page renders (`pageX_imgY`); logs them as skipped instead of saving.

Features
- Caption‑aware naming: picks closest caption above/below and sanitizes to a safe filename.
- Rich logs: `[SAVED]` for stored images and `[SKIP]` for non‑captioned or fallback outputs.
- Works on Windows/macOS/Linux; optional GUI for simple batch runs.

Requirements
- Python 3.9+
- Dependencies: `pymupdf` (PyMuPDF). `Pillow` is optional and used to convert non‑PNG images to PNG when available.
- External tools (recommended): Poppler `pdftohtml` (and `pdftoppm` for fallback). Optional: `qpdf` or `gs` to repair broken PDFs.

Install
- Create a venv and install dependencies:
  - `pip install pymupdf pillow`  (Pillow optional but recommended)
- Install Poppler binaries:
  - Windows: download a Poppler build (e.g., from conda‑forge or community builds) and ensure `pdftohtml.exe` is on `PATH`.
  - macOS: `brew install poppler`
  - Linux: `apt-get install poppler-utils` or your distro equivalent.

Usage (CLI)
- Extract from a single PDF or a directory of PDFs:
  - `python main.py <input.pdf> --output <out_dir>`
  - `python main.py <input_dir> --output <out_dir> [--recursive]`
- Options:
  - `--proximity N` controls caption distance matching (default 80 PDF points).
  - `--recursive` searches for PDFs under the input directory.

Usage (GUI)
- `python main.py --gui` then select input/output and proximity in the window.

Logging
- Saved: `[SAVED] file.pdf page P image I -> Title.png (caption: …)`
- Skipped (no caption): `[SKIP] file.pdf page P image I: no caption within proximity=…`
- Fallback page rasterization is disabled from saving; it logs as skipped `pageX_imgY`.

Output Rules
- A subfolder per PDF is created under the chosen output directory.
- Filenames are derived from captions and sanitized; duplicates get `_2`, `_3`, … suffixes.

Notes On Caption Matching
- Captions start with `Fig`, `Figure`, or `图/圖` (case‑insensitive) and can span multiple text segments/columns.
- Lines with multiple captions (e.g., side‑by‑side figures) are split so each caption is treated independently.

Troubleshooting
- “No PDF files found to process.”: Check input path and file extensions.
- “Required tool 'pdftohtml' not found”: Install Poppler or add it to `PATH`. The tool is strongly recommended for best results.
- Corrupted PDFs: If available, `qpdf` or `gs` is attempted automatically to repair before extraction.

Releases
- Poppler Binaries
  - The app detects tools via `PATH`. To make releases portable, bundle Poppler executables with the release and add them to `PATH` at runtime.
  - Recommended layout in your release archive:
    - `bin/` containing `pdftohtml` and `pdftoppm` (and dependent DLLs on Windows).
    - A launcher script or wrapper that prepends `bin` to `PATH` before invoking Python.
  - Windows specifics:
    - Include `bin\pdftohtml.exe`, `bin\pdftoppm.exe`, and required DLLs next to them.
    - In your launcher `run.cmd`: `set PATH=%~dp0bin;%PATH%` then `python main.py ...`.
  - macOS/Linux specifics:
    - Include Poppler binaries under `bin/` and use a shell wrapper to export `PATH="$(dirname "$0")/bin:$PATH"` before running.

Build Suggestions (optional)
- PyInstaller/py2app can be used to package the Python app; make sure to also ship Poppler in an adjacent `bin/` folder and adjust `PATH` in a small bootstrap script.
- Do not rely on `sips` on macOS; the code uses Pillow if present and otherwise preserves original image formats.

Repository Layout
- `main.py` — cross‑platform CLI/GUI and extraction logic (Poppler first, PyMuPDF fallback).
- `main_macos.py` — macOS‑focused CLI variant using the same extraction approach and no `sips` requirement.
- `data/` — place sample PDFs here for testing.
- `output/` — default output root for extracted images.

License
- See repository terms. If absent, treat as internal usage only.

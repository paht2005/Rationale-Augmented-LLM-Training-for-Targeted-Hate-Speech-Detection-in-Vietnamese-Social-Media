#!/usr/bin/env python3
"""Extract text from Paper_Article CheckList.pdf"""
import os, sys

PDF = os.path.join(os.path.dirname(__file__), "..", "Paper_Article CheckList.pdf")
OUT = os.path.join(os.path.dirname(__file__), "checklist_extracted.txt")

# Try pypdf first
try:
    from pypdf import PdfReader
    reader = PdfReader(PDF)
    text = ""
    for i, page in enumerate(reader.pages):
        text += f"\n--- PAGE {i+1} ---\n"
        text += (page.extract_text() or "")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n[Saved to {OUT}]")
    sys.exit(0)
except ImportError:
    print("pypdf not available, trying pdfplumber...")

# Try pdfplumber
try:
    import pdfplumber
    text = ""
    with pdfplumber.open(PDF) as pdf:
        for i, page in enumerate(pdf.pages):
            text += f"\n--- PAGE {i+1} ---\n"
            text += (page.extract_text() or "")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n[Saved to {OUT}]")
    sys.exit(0)
except ImportError:
    print("pdfplumber not available, trying pdftotext...")

# Try pdftotext CLI
import subprocess
result = subprocess.run(["pdftotext", PDF, "-"], capture_output=True, text=True)
if result.returncode == 0:
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(result.stdout)
    print(result.stdout)
    print(f"\n[Saved to {OUT}]")
else:
    print(f"pdftotext failed: {result.stderr}")
    print("Install: pip install pypdf  OR  pip install pdfplumber  OR  brew install poppler")

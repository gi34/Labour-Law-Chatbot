from pathlib import Path
import sys
import pdfplumber, fitz


def extract_text_from_pdf(pdf_path: Path) -> str:
    # Try direct text extraction first.
    try:
        with fitz.open(pdf_path) as doc:
            text = "\n".join(page.get_text() for page in doc)
            if text.strip():
                return text
    except Exception:
        pass

    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            if text.strip():
                return text
    except Exception:
        pass



    # Fallback to OCR if direct extraction fails.
    try:
        import pytesseract
        from pdf2image import convert_from_path

        images = convert_from_path(str(pdf_path))
        text = "\n".join(pytesseract.image_to_string(image, lang="eng") for image in images)
        return text
    except Exception as exc:
        raise RuntimeError(
            "Failed to extract text. Install fitz (PyMuPDF), pdfplumber, or pytesseract/pdf2image."
        ) from exc


def main() -> int:
    cwd = Path(__file__).resolve().parent
    pdf_path = cwd / "Akta Kerja 1955 (Akta 265).pdf"
    output_path = cwd / "Labour Law.txt"

    if not pdf_path.exists():
        print(f"PDF file not found: {pdf_path}", file=sys.stderr)
        return 1

    text = extract_text_from_pdf(pdf_path)
    output_path.write_text(text, encoding="utf-8")
    print(f"Saved OCR output to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

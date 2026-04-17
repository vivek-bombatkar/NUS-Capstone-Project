import io


def parse_uploaded_file(uploaded_file) -> tuple[str, str]:
    """
    Parse a Streamlit UploadedFile object into (filename, text_content).

    Supports:
    - .txt  — decoded as UTF-8
    - .pdf  — text extracted page by page using pypdf

    Returns a (source_name, content) tuple compatible with the
    DOCUMENTS format used throughout rag_agent.py.
    """
    filename = uploaded_file.name
    file_bytes = uploaded_file.read()

    if filename.endswith(".txt"):
        try:
            content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = file_bytes.decode("latin-1")
        return (filename, content)

    elif filename.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text.strip())
            content = "\n\n".join(pages)
            return (filename, content)
        except Exception as e:
            return (filename, f"[Could not parse PDF: {e}]")

    else:
        return (filename, "[Unsupported file type. Please upload .txt or .pdf]")
import os
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import time
import re
from pypdf import PdfReader
from markdownify import markdownify as md

# Configuration
MERMAID_URLS = [
    "https://mermaid.js.org/intro/",
    "https://mermaid.js.org/syntax/flowchart.html",
    "https://mermaid.js.org/syntax/sequenceDiagram.html",
    "https://mermaid.js.org/syntax/classDiagram.html",
    "https://mermaid.js.org/syntax/stateDiagram.html",
    "https://mermaid.js.org/syntax/entityRelationshipDiagram.html",
    "https://mermaid.js.org/syntax/userJourney.html",
    "https://mermaid.js.org/syntax/gantt.html",
    "https://mermaid.js.org/syntax/pie.html",
    "https://mermaid.js.org/syntax/quadrantChart.html",
    "https://mermaid.js.org/syntax/requirementDiagram.html",
    "https://mermaid.js.org/ecosystem/mermaid-chart.html",
    "https://mermaid.js.org/config/configuration.html",
    "https://mermaid.js.org/config/directives.html",
    "https://mermaid.js.org/config/theming.html",
    "https://mermaid.js.org/community/intro.html"
]

MERMAID_GITHUB_URLS = [
    # Useful for version-specific notes, quick-start snippets, and real-world gotchas
    "https://github.com/mermaid-js/mermaid"
]

UML_DIAGRAMS_URLS = [
    "https://www.uml-diagrams.org/",
    "https://www.uml-diagrams.org/class-diagrams-overview.html",
    "https://www.uml-diagrams.org/component-diagrams.html",
    "https://www.uml-diagrams.org/deployment-diagrams-overview.html",
    "https://www.uml-diagrams.org/use-case-diagrams.html",
    "https://www.uml-diagrams.org/sequence-diagrams.html",
    "https://www.uml-diagrams.org/activity-diagrams.html",
    "https://www.uml-diagrams.org/state-machine-diagrams.html",
    # The old object-diagrams overview page is gone; this example page still exists.
    "https://www.uml-diagrams.org/online-shopping-user-login-uml-object-diagram-example.html",
    "https://www.uml-diagrams.org/package-diagrams-overview.html",
    "https://www.uml-diagrams.org/profile-diagrams.html",
    "https://www.uml-diagrams.org/composite-structure-diagrams.html",
    "https://www.uml-diagrams.org/communication-diagrams.html",
    "https://www.uml-diagrams.org/interaction-overview-diagrams.html",
    "https://www.uml-diagrams.org/timing-diagrams.html",
    # Good index pages that are stable and cover many "core topics"
    "https://www.uml-diagrams.org/uml-25-diagrams.html",
    "https://www.uml-diagrams.org/index-examples.html"
]

PLANTUML_URLS = [
    # Core entry point + common UML diagram types
    "https://plantuml.com/",
    "https://plantuml.com/sequence-diagram",
    "https://plantuml.com/class-diagram",
    "https://plantuml.com/state-diagram",
    "https://plantuml.com/activity-diagram-beta",
    "https://plantuml.com/use-case-diagram",
    "https://plantuml.com/component-diagram",
    "https://plantuml.com/deployment-diagram",
    "https://plantuml.com/object-diagram",
    "https://plantuml.com/timing-diagram",
    # Frequently-needed language features
    "https://plantuml.com/preprocessing",
    "https://plantuml.com/skinparam",
    "https://plantuml.com/theme",
    "https://plantuml.com/stdlib",
    "https://plantuml.com/sprite",
    "https://plantuml.com/creole",
]

PLANTUML_STDLIB_URLS = [
    # Stdlib + C4 patterns (very common in industry PlantUML usage)
    "https://github.com/plantuml/plantuml-stdlib",
    "https://raw.githubusercontent.com/plantuml/plantuml-stdlib/master/C4/README.md",
]

PLANTUML_PDF_PATH = os.path.join(os.getcwd(), "docs", "PlantUML_Language_Reference_Guide_en.pdf")
OMG_UML_PDF_PATH = os.path.join(os.getcwd(), "docs", "OMG® Unified Modeling Language® (OMG UML®)Version 2.5.1.pdf")

# Embedding config
# NOTE: Large embedding models (e.g. 8B) will not fit on a 3GB GPU and often won't fit in RAM either.
# Use env vars to override safely.
MODEL_NAME = os.getenv("UMLGENIE_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
FALLBACK_MODEL_NAME = os.getenv("UMLGENIE_EMBED_FALLBACK_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_DEVICE = os.getenv("UMLGENIE_EMBED_DEVICE", "auto")  # auto|cpu|cuda
EMBED_BATCH_SIZE = int(os.getenv("UMLGENIE_EMBED_BATCH_SIZE", "8"))

BINS_DIR = 'bins'

# Helps reduce CUDA fragmentation when CUDA is explicitly used.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _resolve_device(device_pref="auto", min_cuda_gb=8.0):
    """
    Decide where embeddings should run.
    - 'cpu': force CPU
    - 'cuda': force CUDA (may OOM)
    - 'auto': CUDA only if available and has enough total VRAM
    """
    pref = (device_pref or "auto").strip().lower()
    if pref in ("cpu", "cuda"):
        return pref

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_gb = float(props.total_memory) / (1024 ** 3)
            if total_gb >= float(os.getenv("UMLGENIE_MIN_CUDA_GB", str(min_cuda_gb))):
                return "cuda"
    except Exception:
        pass
    return "cpu"

def _load_embed_model(model_name, device):
    """
    Load SentenceTransformer robustly.
    Falls back to CPU or a smaller model if the requested one cannot be loaded.
    """
    try:
        print(f"Loading embedding model: {model_name} (device={device}) ...")
        return SentenceTransformer(model_name, trust_remote_code=True, device=device)
    except Exception as e:
        msg = str(e).lower()
        is_oom = ("out of memory" in msg) or ("cuda out of memory" in msg)
        if is_oom and device == "cuda":
            print(f"[WARN] CUDA OOM while loading '{model_name}'. Falling back to CPU.")
            return SentenceTransformer(model_name, trust_remote_code=True, device="cpu")
        if model_name != FALLBACK_MODEL_NAME:
            print(f"[WARN] Failed to load '{model_name}'. Falling back to '{FALLBACK_MODEL_NAME}'.")
            return SentenceTransformer(FALLBACK_MODEL_NAME, trust_remote_code=True, device="cpu")
        raise

def _encode_with_backoff(model, texts, batch_size):
    """
    Encode with adaptive batch size and CUDA->CPU fallback on OOM.
    """
    bs = max(1, int(batch_size or 1))
    while True:
        try:
            return model.encode(texts, show_progress_bar=True, batch_size=bs)
        except Exception as e:
            msg = str(e).lower()
            is_oom = ("out of memory" in msg) or ("cuda out of memory" in msg)
            if not is_oom:
                raise
            if bs > 1:
                bs = max(1, bs // 2)
                print(f"[WARN] OOM during encode. Retrying with smaller batch_size={bs} ...")
                continue
            # batch size already 1: try CPU fallback if we're on CUDA
            try:
                if getattr(model, "device", None) is not None and "cuda" in str(model.device).lower():
                    print("[WARN] OOM even at batch_size=1 on CUDA. Moving model to CPU and retrying...")
                    model.to("cpu")
                    return model.encode(texts, show_progress_bar=True, batch_size=1)
            except Exception:
                pass
            raise

class BaseIngestor:
    def __init__(self, name):
        self.name = name

    def ingest(self):
        raise NotImplementedError

    def clean_text(self, text):
        # Basic cleanup
        return re.sub(r'\n{3,}', '\n\n', text).strip()

class WebIngestor(BaseIngestor):
    def __init__(self, name, urls):
        super().__init__(name)
        self.urls = urls
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def scrape_url(self, url):
        try:
            print(f"[{self.name}] Scraping {url}...")
            response = requests.get(url, headers=self.headers, timeout=15)

            # Fix common 404 patterns (especially uml-diagrams.org legacy "*-overview.html" pages)
            if response.status_code == 404:
                retries = []
                if "uml-diagrams.org" in url:
                    if url.endswith("-overview.html"):
                        retries.append(url.replace("-overview.html", ".html"))
                        retries.append(url.replace("-diagrams-overview.html", "-diagrams.html"))
                        retries.append(url.replace("-diagram-overview.html", "-diagram.html"))
                    # Some old "index" pages were removed; keep a short set of known-good fallbacks
                    if url.endswith("uml-core-topics-index.html"):
                        retries.extend([
                            "https://www.uml-diagrams.org/uml-25-diagrams.html",
                            "https://www.uml-diagrams.org/index-examples.html",
                        ])

                for alt in retries:
                    if alt == url:
                        continue
                    try:
                        print(f"[{self.name}] 404 -> retrying {alt} ...")
                        response = requests.get(alt, headers=self.headers, timeout=15)
                        if response.status_code != 404:
                            url = alt  # keep correct source URL
                            break
                    except Exception:
                        continue

            response.raise_for_status()
            content_type = (response.headers.get("Content-Type") or "").lower()

            # Handle raw text sources (e.g., raw GitHub markdown, .puml files, etc.)
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return self.clean_text(response.text or "")

            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove clutter
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                tag.decompose()

            # Main content usually in 'main' or 'article' or specific divs
            # For mermaid docs, it's often within an article or main
            if "github.com" in url:
                content = (
                    soup.select_one("#readme article.markdown-body")
                    or soup.select_one("article.markdown-body")
                    or soup.find("main")
                    or soup.find("article")
                    or soup.body
                )
            else:
                content = (
                    soup.find("main")
                    or soup.find("article")
                    or soup.find("div", id="content")
                    or soup.body
                )
            
            if content:
                html_content = str(content)
                text = md(html_content, heading_style="ATX")
                return self.clean_text(text)
            return ""
        except Exception as e:
            print(f"[{self.name}] Failed to scrape {url}: {e}")
            return ""

    def ingest(self):
        chunks = []
        for url in self.urls:
            text = self.scrape_url(url)
            if text:
                file_chunks = split_text(text)
                chunks.extend(attach_chunk_metadata(
                    file_chunks,
                    source=url,
                    topic=self.name,
                    doc_type="web",
                ))
            time.sleep(1) # Be polite
        return chunks

class PDFIngestor(BaseIngestor):
    def __init__(self, name, pdf_path):
        super().__init__(name)
        self.pdf_path = pdf_path

    def ingest(self):
        chunks = []
        if not os.path.exists(self.pdf_path):
            print(f"[{self.name}] PDF not found: {self.pdf_path}")
            return chunks

        try:
            print(f"[{self.name}] Reading PDF: {self.pdf_path}...")
            reader = PdfReader(self.pdf_path)
            full_text = ""
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    full_text += f"\n\n--- Page {i+1} ---\n\n" + text
            
            cleaned_text = self.clean_text(full_text)
            raw_chunks = split_text(cleaned_text)
            chunks.extend(attach_chunk_metadata(
                raw_chunks,
                source=os.path.basename(self.pdf_path),
                topic=self.name,
                doc_type="pdf",
            ))
        except Exception as e:
            print(f"[{self.name}] Error reading PDF: {e}")
            
        return chunks

class CompositeIngestor(BaseIngestor):
    """
    Combines multiple ingestors into a single *branch* so the saved index name remains stable.
    Example: keep everything PlantUML-related under the 'plantuml' branch (PDF + official site + stdlib).
    """
    def __init__(self, name, ingestors):
        super().__init__(name)
        self.ingestors = ingestors

    def ingest(self):
        all_chunks = []
        for ing in self.ingestors:
            try:
                all_chunks.extend(ing.ingest())
            except Exception as e:
                print(f"[{self.name}] Sub-ingestor '{getattr(ing, 'name', 'unknown')}' failed: {e}")
        return all_chunks

def _update_heading_stack(stack, level, title):
    """
    Maintain a breadcrumb-like heading stack.
    level: 1..6
    """
    if level <= 0:
        return stack
    stack = list(stack[: max(0, level - 1)])
    stack.append(title.strip())
    return stack

def split_text(text, chunk_size=1600, chunk_overlap=200):
    """
    Smarter chunking that:
    - Preserves fenced code blocks (``` ... ```) so Mermaid/PlantUML snippets aren't split mid-block.
    - Retains section breadcrumbs (headings) as a lightweight prefix to improve retrieval accuracy.
    - Falls back to character chunking for very large blocks.

    Returns: list[str]
    """
    if not text:
        return []

    # Quick path for PDFs that contain page markers
    if "--- Page " in text and " ---" in text:
        pages = re.split(r"\n\s*--- Page \d+ ---\s*\n", text)
        pages = [p.strip() for p in pages if p.strip()]
        out = []
        for p in pages:
            out.extend(split_text(p, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
        return out

    lines = text.splitlines()
    blocks = []  # (kind, heading_stack_tuple, block_text)
    heading_stack = tuple()
    buf = []
    in_code = False
    code_lang = ""

    def flush(kind):
        nonlocal buf, blocks
        if not buf:
            return
        block_text = "\n".join(buf).strip("\n")
        if block_text.strip():
            blocks.append((kind, heading_stack, block_text))
        buf = []

    for line in lines:
        # Code fences: treat as atomic blocks
        if line.strip().startswith("```"):
            if not in_code:
                flush("text")
                in_code = True
                code_lang = line.strip()[3:].strip()
                buf.append(line)
            else:
                buf.append(line)
                flush(f"code:{code_lang or 'plain'}")
                in_code = False
                code_lang = ""
            continue

        if in_code:
            buf.append(line)
            continue

        # Markdown heading
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if m:
            flush("text")
            level = len(m.group(1))
            title = m.group(2)
            heading_stack = tuple(_update_heading_stack(heading_stack, level, title))
            blocks.append(("heading", heading_stack, f"{m.group(1)} {title}".strip()))
            continue

        buf.append(line)

    flush("code" if in_code else "text")

    # Combine blocks into chunks with lightweight overlap
    chunks = []
    cur = []
    cur_len = 0
    last_heading = tuple()

    def heading_prefix(hstack):
        if not hstack:
            return ""
        return "Section: " + " > ".join(hstack)

    def flush_chunk():
        nonlocal cur, cur_len, last_heading
        if not cur:
            return
        prefix = heading_prefix(last_heading)
        body = "\n\n".join(cur).strip()
        chunk = (prefix + "\n\n" + body).strip() if prefix else body
        if chunk:
            chunks.append(chunk)
        cur = []
        cur_len = 0

    for kind, hstack, block_text in blocks:
        if hstack:
            last_heading = hstack

        # If a single block is huge, split it.
        if len(block_text) > chunk_size * 2:
            flush_chunk()
            start = 0
            while start < len(block_text):
                end = start + chunk_size
                part = block_text[start:end]
                prefix = heading_prefix(hstack)
                part = (prefix + "\n\n" + part).strip() if prefix else part.strip()
                chunks.append(part)
                start += max(1, chunk_size - chunk_overlap)
            continue

        block_len = len(block_text) + 2
        if cur_len + block_len > chunk_size and cur:
            flush_chunk()
            # Overlap tail characters (helps avoid missing surrounding context)
            if chunk_overlap and chunks:
                tail = chunks[-1][-chunk_overlap:].strip()
                if tail:
                    cur = [tail]
                    cur_len = len(tail)

        cur.append(block_text)
        cur_len += block_len

    flush_chunk()
    return chunks

def attach_chunk_metadata(chunks, source, topic, doc_type):
    """
    Convert list[str] -> list[dict] with stable metadata so RAG can show section/source info
    and optionally return neighboring chunks.
    """
    out = []
    for i, c in enumerate(chunks):
        section = ""
        if c.startswith("Section: "):
            first_line, _, _rest = c.partition("\n")
            section = first_line.replace("Section:", "", 1).strip()
        out.append({
            "text": c,
            "source": source,
            "topic": topic,
            "type": doc_type,
            "doc_id": source,
            "chunk_idx": i,
            "section": section,
        })
    return out

def process_branch(branch_name, data_chunks, model):
    if not data_chunks:
        print(f"[{branch_name}] No data to process.")
        return

    print(f"[{branch_name}] Encoding {len(data_chunks)} chunks...")
    texts = [c['text'] for c in data_chunks]
    embeddings = _encode_with_backoff(model, texts, batch_size=EMBED_BATCH_SIZE)
    embeddings = np.asarray(embeddings, dtype="float32")
    # Normalize for cosine similarity with inner-product index.
    faiss.normalize_L2(embeddings)
    
    print(f"[{branch_name}] Building Index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    index_file = os.path.join(BINS_DIR, f"{branch_name.lower().replace(' ', '_')}_index.bin")
    chunks_file = os.path.join(BINS_DIR, f"{branch_name.lower().replace(' ', '_')}_chunks.pkl")
    
    faiss.write_index(index, index_file)
    with open(chunks_file, 'wb') as f:
        pickle.dump(data_chunks, f)
    print(f"[{branch_name}] Saved to {index_file}")

def main():
    if not os.path.exists(BINS_DIR):
        os.makedirs(BINS_DIR)
        
    device = _resolve_device(EMBED_DEVICE)
    model = _load_embed_model(MODEL_NAME, device=device)

    # Define Branches
    branches = [
        WebIngestor("Mermaid", MERMAID_URLS + MERMAID_GITHUB_URLS),
        WebIngestor("General UML", UML_DIAGRAMS_URLS),
        CompositeIngestor("PlantUML", [
            WebIngestor("PlantUML Web", PLANTUML_URLS),
            WebIngestor("PlantUML Stdlib", PLANTUML_STDLIB_URLS),
            PDFIngestor("PlantUML PDF", PLANTUML_PDF_PATH),
        ]),
        PDFIngestor("OMG UML", OMG_UML_PDF_PATH)
    ]

    for branch in branches:
        print(f"\n--- Processing Branch: {branch.name} ---")
        chunks = branch.ingest()
        process_branch(branch.name, chunks, model)

if __name__ == "__main__":
    main()

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
    "https://mermaid.js.org/community/intro.html"
]

UML_DIAGRAMS_URLS = [
    "https://www.uml-diagrams.org/",
    "https://www.uml-diagrams.org/class-diagrams-overview.html",
    "https://www.uml-diagrams.org/component-diagrams-overview.html",
    "https://www.uml-diagrams.org/deployment-diagrams-overview.html",
    "https://www.uml-diagrams.org/use-case-diagrams-overview.html",
    "https://www.uml-diagrams.org/sequence-diagrams-overview.html",
    "https://www.uml-diagrams.org/activity-diagrams-overview.html",
    "https://www.uml-diagrams.org/state-machine-diagrams-overview.html",
    "https://www.uml-diagrams.org/object-diagrams-overview.html",
    "https://www.uml-diagrams.org/package-diagrams-overview.html",
    "https://www.uml-diagrams.org/profile-diagrams-overview.html",
    "https://www.uml-diagrams.org/composite-structure-diagrams.html",
    "https://www.uml-diagrams.org/communication-diagrams-overview.html",
    "https://www.uml-diagrams.org/interaction-overview-diagrams.html",
    "https://www.uml-diagrams.org/timing-diagrams-overview.html",
    "https://www.uml-diagrams.org/uml-core-topics-index.html"
]

PLANTUML_PDF_PATH = os.path.join(os.getcwd(), "docs", "PlantUML_Language_Reference_Guide_en.pdf")
OMG_UML_PDF_PATH = os.path.join(os.getcwd(), "docs", "OMG® Unified Modeling Language® (OMG UML®)Version 2.5.1.pdf")

MODEL_NAME = 'Qwen/Qwen3-Embedding-0.6B'
BINS_DIR = 'bins'

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
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove clutter
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                tag.decompose()

            # Main content usually in 'main' or 'article' or specific divs
            # For mermaid docs, it's often within an article or main
            content = soup.find('main') or soup.find('article') or soup.body
            
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
                for c in file_chunks:
                    chunks.append({
                        'text': c,
                        'source': url,
                        'topic': self.name,
                        'type': 'web'
                    })
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
            
            for rc in raw_chunks:
                chunks.append({
                    'text': rc,
                    'source': os.path.basename(self.pdf_path),
                    'topic': self.name,
                    'type': 'pdf'
                })
        except Exception as e:
            print(f"[{self.name}] Error reading PDF: {e}")
            
        return chunks

def split_text(text, chunk_size=800, chunk_overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def process_branch(branch_name, data_chunks, model):
    if not data_chunks:
        print(f"[{branch_name}] No data to process.")
        return

    print(f"[{branch_name}] Encoding {len(data_chunks)} chunks...")
    texts = [c['text'] for c in data_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=16)
    
    print(f"[{branch_name}] Building Index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    index_file = os.path.join(BINS_DIR, f"{branch_name.lower().replace(' ', '_')}_index.bin")
    chunks_file = os.path.join(BINS_DIR, f"{branch_name.lower().replace(' ', '_')}_chunks.pkl")
    
    faiss.write_index(index, index_file)
    with open(chunks_file, 'wb') as f:
        pickle.dump(data_chunks, f)
    print(f"[{branch_name}] Saved to {index_file}")

def main():
    if not os.path.exists(BINS_DIR):
        os.makedirs(BINS_DIR)
        
    print(f"Loading Model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    # Define Branches
    branches = [
        WebIngestor("Mermaid", MERMAID_URLS),
        WebIngestor("General UML", UML_DIAGRAMS_URLS),
        PDFIngestor("PlantUML", PLANTUML_PDF_PATH),
        PDFIngestor("OMG UML", OMG_UML_PDF_PATH)
    ]

    for branch in branches:
        print(f"\n--- Processing Branch: {branch.name} ---")
        chunks = branch.ingest()
        process_branch(branch.name, chunks, model)

if __name__ == "__main__":
    main()

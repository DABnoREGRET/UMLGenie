import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import glob

class UMLRAG:
    def __init__(self, bins_dir='bins', model_name='Qwen/Qwen3-Embedding-0.6B'):
        self.bins_dir = bins_dir
        self.model_name = model_name
        self.indices = {} # {'mermaid': index, 'plantuml': index, ...}
        self.chunks = {}  # {'mermaid': [chunks], ...}
        self.model = None

    def load(self):
        try:
            print(f"Loading RAG model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
            
            # Find all index files
            index_files = glob.glob(os.path.join(self.bins_dir, '*_index.bin'))
            
            if not index_files:
                print("No RAG indices found. Please run ingest_docs.py first.")
                return False

            for idx_path in index_files:
                # Extract branch name usually 'mermaid_index.bin' -> 'mermaid'
                filename = os.path.basename(idx_path)
                branch_name = filename.replace('_index.bin', '')
                
                chunks_path = os.path.join(self.bins_dir, f"{branch_name}_chunks.pkl")
                
                if os.path.exists(chunks_path):
                    print(f"Loading branch: {branch_name}")
                    self.indices[branch_name] = faiss.read_index(idx_path)
                    with open(chunks_path, 'rb') as f:
                        self.chunks[branch_name] = pickle.load(f)
            
            return True
        except Exception as e:
            print(f"Error loading RAG: {e}")
            return False

    def route_query(self, query_text):
        """
        Semantic Classifier using the embedding model.
        Matches query against branch descriptions.
        """
        if not self.model:
            return "mermaid" # Fallback
            
        # Branch Descriptions - Enriched for Syntax/Examples
        candidates = {
            "mermaid": "mermaid js diagram syntax flowchart sequence class state chart code examples usage guide tutorial",
            "plantuml": "plantuml diagram syntax component deployment usecase class code examples usage guide tutorial reference",
            "omg_uml": "omg uml specification standard definitions metamodel syntax rules compliance examples",
            "general_uml": "general uml theory design patterns object oriented analysis examples best practices"
        }
        
        # Embed query
        query_emb = self.model.encode(query_text)
        
        best_branch = "mermaid"
        max_sim = -1.0
        
        for branch, desc in candidates.items():
            desc_emb = self.model.encode(desc)
            # Cosine similarity
            sim = np.dot(query_emb, desc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(desc_emb))
            
            if sim > max_sim:
                max_sim = sim
                best_branch = branch
                
        print(f"🔀 Semantic Router: '{query_text}' -> {best_branch} (score: {max_sim:.2f})")
        return best_branch

    def query(self, query_text, k=5, branch=None):
        if not self.indices or not self.model:
            success = self.load()
            if not success:
                return []
        
        # Determine branch if not specified
        target_branch = branch
        if not target_branch:
            target_branch = self.route_query(query_text)
            
        # Fallback if branch doesn't exist
        if target_branch not in self.indices:
            print(f"Branch '{target_branch}' not found. Available: {list(self.indices.keys())}")
            # Identify first available or return empty
            if self.indices:
                target_branch = list(self.indices.keys())[0]
            else:
                return []

        print(f"Querying Branch: {target_branch}")
        
        vector = self.model.encode([query_text])
        index = self.indices[target_branch]
        branch_chunks = self.chunks[target_branch]
        
        distances, indices = index.search(np.array(vector), k)
        
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(branch_chunks):
                chunk_data = branch_chunks[idx]
                formatted_result = f"[Topic: {chunk_data['topic']}] [Source: {chunk_data['source']}]\n{chunk_data['text']}"
                results.append(formatted_result)
                
        return results

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import glob

class UMLRAG:
    def __init__(self, bins_dir='bins', model_name=None, device=None):
        self.bins_dir = bins_dir
        self.model_name = model_name or os.getenv("UMLGENIE_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
        self.device = device or os.getenv("UMLGENIE_EMBED_DEVICE", "auto")
        self.indices = {} # {'mermaid': index, 'plantuml': index, ...}
        self.chunks = {}  # {'mermaid': [chunks], ...}
        self.model = None
        self._route_candidates = None
        self._route_candidate_matrix = None

    def load(self):
        try:
            device = self._resolve_device(self.device)
            print(f"Loading RAG model: {self.model_name} (device={device})...")
            self.model = SentenceTransformer(self.model_name, trust_remote_code=True, device=device)
            
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

            # Prepare router candidates once (avoid re-embedding per query)
            self._prepare_router()
            return True
        except Exception as e:
            print(f"Error loading RAG: {e}")
            return False

    def _resolve_device(self, device_pref="auto", min_cuda_gb=3.0):
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

    def _prepare_router(self):
        """
        Precompute embeddings for branch descriptions used by the semantic router.
        """
        candidates = {
            "mermaid": "mermaid js diagram syntax flowchart graph TD sequenceDiagram classDiagram stateDiagram erDiagram gantt pie directives init theming tutorial examples",
            "plantuml": "plantuml diagram syntax @startuml @enduml class sequence activity state usecase component deployment skinparam theme preprocessing include stdlib C4 examples reference",
            "omg_uml": "omg uml specification standard definitions metamodel semantics compliance",
            "general_uml": "general uml theory relationships association aggregation composition multiplicity design patterns best practices",
        }
        self._route_candidates = candidates
        if not self.model:
            self._route_candidate_matrix = None
            return
        descs = list(candidates.values())
        mat = self.model.encode(descs)
        mat = np.asarray(mat, dtype="float32")
        faiss.normalize_L2(mat)
        self._route_candidate_matrix = mat

    def route_query(self, query_text):
        """
        Semantic Classifier using the embedding model.
        Matches query against branch descriptions.
        """
        if not self.model:
            return "mermaid" # Fallback

        q = (query_text or "").lower()
        # High-precision keyword routing first (avoids semantic misses)
        if "plantuml" in q or "@startuml" in q or "@enduml" in q:
            return "plantuml"
        if "mermaid" in q or "sequencediagram" in q or "classdiagram" in q or "flowchart" in q or "graph td" in q or "erdiagram" in q:
            return "mermaid"
        if "omg" in q and "uml" in q:
            return "omg_uml"

        # Semantic fallback
        if self._route_candidate_matrix is None or self._route_candidates is None:
            self._prepare_router()
        if self._route_candidate_matrix is None:
            return "mermaid"

        query_emb = self.model.encode([query_text])
        query_emb = np.asarray(query_emb, dtype="float32")
        faiss.normalize_L2(query_emb)

        # Cosine sim via dot product
        sims = np.dot(self._route_candidate_matrix, query_emb[0])
        branches = list(self._route_candidates.keys())
        best_i = int(np.argmax(sims))
        best_branch = branches[best_i]
        print(f"🔀 Semantic Router: '{query_text}' -> {best_branch} (score: {float(sims[best_i]):.2f})")
        return best_branch

    def _search_branch(self, branch_name, query_text, k):
        """
        Returns list of tuples: (score, branch_name, idx)
        score is higher-is-better (cosine sim when possible, otherwise negative L2 distance).
        """
        if branch_name not in self.indices:
            return []
        index = self.indices[branch_name]
        vector = self.model.encode([query_text])
        vector = np.asarray(vector, dtype="float32")

        # Try to match the indexing strategy (cosine sim for IndexFlatIP with normalized vectors).
        try:
            faiss.normalize_L2(vector)
        except Exception:
            pass

        distances, indices = index.search(vector, k)
        out = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            # If index is L2, smaller distance is better. Convert to score.
            score = float(dist)
            if getattr(index, "metric_type", None) == faiss.METRIC_L2:
                score = -float(dist)
            out.append((score, branch_name, int(idx)))
        return out

    def _expand_neighbors(self, branch_name, hit_indices, radius=1):
        """
        Given a set of chunk indices, include neighbors (idx-1..idx+1) when chunk metadata supports it.
        """
        if radius <= 0:
            return set(hit_indices)
        if branch_name not in self.chunks:
            return set(hit_indices)

        # If chunks don't have 'chunk_idx', we still can expand by raw list index adjacency.
        expanded = set()
        n = len(self.chunks[branch_name])
        for idx in hit_indices:
            for j in range(max(0, idx - radius), min(n, idx + radius + 1)):
                expanded.add(j)
        return expanded

    def query(self, query_text, k=5, branch=None, include_neighbors=1, multi_branch_auto=True):
        if not self.indices or not self.model:
            success = self.load()
            if not success:
                return []
        
        # Determine branch(es) if not specified
        target_branch = branch
        branches_to_search = []
        if target_branch:
            branches_to_search = [target_branch]
        else:
            best = self.route_query(query_text)
            branches_to_search = [best]

            # Reduce missed information by also searching a second likely branch
            # when user didn't explicitly ask for a language.
            q = (query_text or "").lower()
            explicitly_scoped = ("plantuml" in q) or ("mermaid" in q) or ("@startuml" in q)
            if multi_branch_auto and not explicitly_scoped:
                # Try a sensible fallback pair
                if best == "mermaid" and "plantuml" in self.indices:
                    branches_to_search.append("plantuml")
                elif best == "plantuml" and "mermaid" in self.indices:
                    branches_to_search.append("mermaid")

        # Filter to branches that actually exist
        branches_to_search = [b for b in branches_to_search if b in self.indices]
        if not branches_to_search:
            print(f"No matching branches found. Available: {list(self.indices.keys())}")
            return []

        # Search and merge
        k_per = max(k, 8)
        scored_hits = []
        for b in branches_to_search:
            scored_hits.extend(self._search_branch(b, query_text, k_per))

        scored_hits.sort(key=lambda x: x[0], reverse=True)

        results = []
        seen = set()
        for score, b, idx in scored_hits:
            if b not in self.chunks:
                continue
            branch_chunks = self.chunks[b]
            if not (0 <= idx < len(branch_chunks)):
                continue

            # Neighbor expansion (helps avoid missing context around a code snippet)
            expanded = self._expand_neighbors(b, [idx], radius=include_neighbors)
            for j in sorted(expanded):
                if (b, j) in seen:
                    continue
                if not (0 <= j < len(branch_chunks)):
                    continue
                chunk_data = branch_chunks[j]
                topic = chunk_data.get("topic", b)
                source = chunk_data.get("source", "")
                section = chunk_data.get("section", "")
                section_part = f" [Section: {section}]" if section else ""
                formatted_result = f"[Branch: {b}] [Topic: {topic}] [Source: {source}]{section_part}\n{chunk_data.get('text','')}"
                results.append(formatted_result)
                seen.add((b, j))
                if len(results) >= k:
                    return results

        return results

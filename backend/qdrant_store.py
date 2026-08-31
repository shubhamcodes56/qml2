"""
Qdrant Vector Store — 3M Quantum States with Scalar Quantization
================================================================
Stores 256-dim complex state vectors (512 floats) with Scalar Quantization
enabled to fit ~3.2 Million vectors in Qdrant Cloud Free Tier (4 GiB disk).

Scalar Quantization compresses Float32 → Int8, reducing vector size by ~4x
while maintaining 99%+ recall accuracy.
"""

import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    ScalarQuantization, ScalarQuantizationConfig, ScalarType,
    OptimizersConfigDiff, HnswConfigDiff,
    Filter, FieldCondition, MatchValue
)

COLLECTION_NAME = "tb_quantum_states"
VECTOR_DIM = 512  # 256 complex → 512 floats


class QuantumVectorStore:
    def __init__(self, url=None, api_key=None, persist_dir=None):
        """
        Connect to Qdrant.
        - url + api_key: Qdrant Cloud (recommended for 3M vectors)
        - persist_dir: Local disk storage
        - Neither: In-memory (testing only)
        """
        if url and api_key:
            self.client = QdrantClient(url=url, api_key=api_key)
            print(f"[QDRANT] Connected to Cloud: {url}")
        elif persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self.client = QdrantClient(path=persist_dir)
            print(f"[QDRANT] Local storage: {persist_dir}")
        else:
            self.client = QdrantClient(":memory:")
            print("[QDRANT] In-memory mode")
        
        self._ensure_collection()
    
    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_DIM,
                    distance=Distance.COSINE,
                    on_disk=True,  # Use disk-backed storage for large collections
                ),
                # Scalar Quantization: Float32 → Int8 (4x compression)
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,  # Keep quantized vectors in RAM for speed
                    )
                ),
                # Optimized for large collections
                optimizers_config=OptimizersConfigDiff(
                    memmap_threshold=20000,
                    indexing_threshold=20000,
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    on_disk=True,
                ),
            )
            print(f"[QDRANT] Created collection '{COLLECTION_NAME}' with Scalar Quantization (Int8)")
    
    def state_to_vector(self, state_vector_complex):
        """Complex 256-dim → Real 512-dim (interleaved real/imag)."""
        state = np.array(state_vector_complex, dtype=np.complex128)
        vec = np.empty(2 * len(state), dtype=np.float32)
        vec[0::2] = state.real
        vec[1::2] = state.imag
        return vec.tolist()
    
    def store_patient(self, patient_id, state_vector, metadata=None):
        vector = self.state_to_vector(state_vector)
        payload = metadata or {}
        payload["patient_id"] = patient_id
        
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(
                id=int(patient_id) if isinstance(patient_id, (int, np.integer)) else abs(hash(patient_id)) % (2**63),
                vector=vector,
                payload=payload,
            )]
        )
    
    def find_similar(self, state_vector, top_k=5, severity_filter=None):
        """Find K most similar patients in quantum state space."""
        query_vec = self.state_to_vector(state_vector)
        
        search_filter = None
        if severity_filter:
            search_filter = Filter(
                must=[FieldCondition(key="severity", match=MatchValue(value=severity_filter))]
            )
        
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=top_k,
            query_filter=search_filter,
        )
        
        return [{
            "patient_id": p.payload.get("patient_id", p.id),
            "similarity": round(p.score, 4),
            "tier": p.payload.get("tier", "?"),
            "risk_score": p.payload.get("risk_score", 0),
            "severity": p.payload.get("severity", "?"),
        } for p in results.points]
    
    def bulk_store(self, patient_ids, state_vectors, metadata_list, batch_size=500):
        """Bulk insert for populating the database."""
        points = []
        for pid, sv, meta in zip(patient_ids, state_vectors, metadata_list):
            vec = self.state_to_vector(sv)
            payload = meta or {}
            payload["patient_id"] = int(pid)
            points.append(PointStruct(id=int(pid), vector=vec, payload=payload))
        
        for i in range(0, len(points), batch_size):
            self.client.upsert(collection_name=COLLECTION_NAME, points=points[i:i+batch_size])
        
        print(f"[QDRANT] Stored {len(points):,} quantum state vectors")
    
    def get_info(self):
        info = self.client.get_collection(COLLECTION_NAME)
        return {
            "total_vectors": info.points_count,
            "vector_dim": VECTOR_DIM,
            "quantization": "Scalar (Int8, 4x compression)",
            "distance": "COSINE",
        }

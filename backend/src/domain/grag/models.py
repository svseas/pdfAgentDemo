import os
import re
import torch
import numpy as np
import networkx as nx
import logging
from typing import List, Dict, Tuple, Optional, Union
import penman
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import GCNConv

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AMRParser:
    """Wrapper for AMR parsing using AMRBART"""
    def __init__(self, model_name="xfbai/AMRBART-large-finetuned-AMR3.0-seq2seq"):
        try:
            logger.info(f"Loading AMR parser model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            logger.info("AMR parser loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load AMR parser: {e}")
            raise

    def parse(self, text: str) -> str:
        try:
            inputs = self.tokenizer(f"question:{text}", return_tensors="pt", padding=True).to(device)
            outputs = self.model.generate(**inputs, max_length=512)
            amr_graph = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return amr_graph
        except Exception as e:
            logger.error(f"Error parsing text to AMR: {e}")
            return ""

    def parse_batch(self, texts: List[str], batch_size: int = 4) -> List[str]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = []
            for text in batch_texts:
                try:
                    batch_results.append(self.parse(text))
                except Exception as e:
                    logger.error(f"Error parsing text in batch: {e}")
                    batch_results.append("")
            results.extend(batch_results)
        return results

class AMRGraphProcessor:
    """Process AMR graphs to extract node concepts and compute paths"""
    def __init__(self):
        self._cache = {}

    def extract_node_concepts(self, amr_graph: str) -> List[str]:
        if not amr_graph or len(amr_graph) < 5:
            return []
            
        cache_key = f"concepts_{hash(amr_graph)}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        try:
            graph = penman.decode(amr_graph)
            concepts = []
            for triple in graph.triples:
                if triple[1] == ':instance':
                    concepts.append(triple[2])
            
            self._cache[cache_key] = concepts
            return concepts
        except Exception as e:
            logger.warning(f"Error extracting concepts: {e}")
            return self._extract_concepts_regex(amr_graph)

    def _extract_concepts_regex(self, amr_graph: str) -> List[str]:
        concepts = []
        matches = re.finditer(r'\(([^\s/()]+)\s*/\s*([^\s()]+)', amr_graph)
        for match in matches:
            if match.group(1) and match.group(1) not in concepts:
                concepts.append(match.group(1))
            if match.group(2) and match.group(2) not in concepts:
                concepts.append(match.group(2))
        return concepts

    def extract_edges(self, amr_graph: str) -> List[Tuple[str, str, str]]:
        if not amr_graph or len(amr_graph) < 5:
            return []
            
        cache_key = f"edges_{hash(amr_graph)}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        try:
            graph = penman.decode(amr_graph)
            var_to_concept = {}
            for triple in graph.triples:
                if triple[1] == ':instance':
                    var_to_concept[triple[0]] = triple[2]
            
            edges = []
            for triple in graph.triples:
                if triple[1] != ':instance':
                    source_var = triple[0]
                    target_var = triple[2]
                    if source_var in var_to_concept and target_var in var_to_concept:
                        source = var_to_concept[source_var]
                        target = var_to_concept[target_var]
                        relation = triple[1].lstrip(':')
                        edges.append((source, relation, target))
            
            self._cache[cache_key] = edges
            return edges
        except Exception as e:
            logger.warning(f"Error extracting edges: {e}")
            return []

    def amr_to_networkx(self, amr_graph: str) -> nx.DiGraph:
        if not amr_graph or len(amr_graph) < 5:
            return nx.DiGraph()
            
        cache_key = f"nx_{hash(amr_graph)}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        G = nx.DiGraph()
        try:
            concepts = self.extract_node_concepts(amr_graph)
            edges = self.extract_edges(amr_graph)
            
            for concept in concepts:
                G.add_node(concept)
            
            for src, rel, tgt in edges:
                G.add_edge(src, tgt, relation=rel)
            
            self._cache[cache_key] = G
            return G
        except Exception as e:
            logger.warning(f"Error converting AMR to NetworkX: {e}")
            return nx.DiGraph()

class GNNReranker(torch.nn.Module):
    """Graph Neural Network for document reranking"""
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super(GNNReranker, self).__init__()
        
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.output_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        
        logger.info(f"GNN Reranker initialized with input_dim={input_dim}, hidden_dim={hidden_dim}")

    def forward(self, x, edge_index, edge_attr):
        if edge_attr.size(0) > 0:
            edge_weights = edge_attr.sum(dim=1)
            edge_weights = torch.nn.functional.normalize(edge_weights, dim=0)
        else:
            edge_weights = edge_attr

        x = self.input_proj(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        if edge_index.size(1) > 0:
            x = torch.relu(self.conv1(x, edge_index, edge_weights))
        else:
            x = torch.relu(x)
        
        x = self.dropout(x)
        
        if edge_index.size(1) > 0:
            x = self.conv2(x, edge_index, edge_weights)
        
        x = self.output_proj(x)
        return x

    def compute_scores(self, node_features, question_embedding):
        if len(question_embedding.shape) == 1:
            question_embedding = question_embedding.unsqueeze(0)
        scores = torch.matmul(node_features, question_embedding.t()).squeeze()
        return scores

class EmbeddingModel:
    """Wrapper for document and question embedding models"""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.model_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded with dimension: {self.model_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def encode(self, text: Union[str, List[str]]) -> torch.Tensor:
        try:
            embeddings = self.model.encode(text, convert_to_tensor=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            if isinstance(text, list):
                return torch.zeros((len(text), self.model_dim))
            else:
                return torch.zeros(self.model_dim)

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings = self.encode(batch)
            all_embeddings.append(embeddings)
        
        if all_embeddings:
            return torch.cat(all_embeddings, dim=0)
        else:
            return torch.zeros((0, self.model_dim))
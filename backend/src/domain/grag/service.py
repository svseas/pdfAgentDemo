import logging
import torch
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Any
from .models import AMRParser, AMRGraphProcessor, GNNReranker, EmbeddingModel

logger = logging.getLogger(__name__)

class GRAGService:
    """Graph-based Reranking Service for RAG Enhancement"""
    
    def __init__(self, 
                embedding_model_name: str = "BAAI/bge-small-en",
                amr_model_name: str = "xfbai/AMRBART-large-finetuned-AMR3.0-seq2seq"):
        
        logger.info("Initializing G-RAG Service...")
        
        # Initialize components
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.amr_parser = AMRParser(amr_model_name)
        self.amr_processor = AMRGraphProcessor()
        
        # Initialize GNN reranker
        self.gnn_reranker = GNNReranker(
            input_dim=self.embedding_model.model_dim,
            hidden_dim=128,
            dropout=0.1
        ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        logger.info("G-RAG Service initialized successfully")

    def build_document_graph(self, question: str, documents: List[Dict[str, Any]]) -> Tuple[nx.Graph, List[str]]:
        """Build document graph based on AMR connections"""
        
        logger.info(f"Building document graph for {len(documents)} documents")
        
        # Create concat texts and parse to AMR
        concat_texts = [f"{question} {doc['content']}" for doc in documents]
        amr_graphs = self.amr_parser.parse_batch(concat_texts)
        
        # Build the document graph
        doc_graph = nx.Graph()
        
        # Add nodes for each document
        for i in range(len(documents)):
            doc_graph.add_node(i)
        
        # Add edges between documents that share common concepts
        edge_count = 0
        for i in range(len(documents)):
            for j in range(i+1, len(documents)):
                common_nodes, common_edges = self.amr_processor.get_common_elements(
                    amr_graphs[i], amr_graphs[j])
                
                if common_nodes:
                    doc_graph.add_edge(
                        i, j,
                        common_nodes=len(common_nodes),
                        common_edges=len(common_edges)
                    )
                    edge_count += 1
        
        logger.info(f"Document graph built with {len(documents)} nodes and {edge_count} edges")
        return doc_graph, amr_graphs

    def prepare_graph_data(self, graph: nx.Graph) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert NetworkX graph to PyTorch tensors for GNN processing"""
        
        edge_index = []
        edge_attr = []
        
        for u, v, data in graph.edges(data=True):
            edge_index.append([u, v])
            edge_index.append([v, u])
            
            attr = [data.get('common_nodes', 0), data.get('common_edges', 0)]
            edge_attr.append(attr)
            edge_attr.append(attr)
        
        device = next(self.gnn_reranker.parameters()).device
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().to(device)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float).to(device)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)
            edge_attr = torch.zeros((0, 2), dtype=torch.float).to(device)
            
        return edge_index, edge_attr

    def rerank(self, question: str, documents: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank documents using G-RAG approach
        
        Args:
            question: User query
            documents: List of document chunks with embeddings
            top_k: Number of documents to return (defaults to all)
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        if len(documents) <= 1:
            return documents
            
        # Build document graph
        doc_graph, amr_graphs = self.build_document_graph(question, documents)
        
        # Extract node features
        node_features = []
        for i, doc in enumerate(documents):
            try:
                # Extract AMR path information
                amr_info = self.amr_processor.get_path_information(amr_graphs[i])
                
                # Concatenate document text with AMR information
                augmented_text = f"{doc['content']} {amr_info}"
                
                # Create embedding
                embedding = self.embedding_model.encode(augmented_text)
                node_features.append(embedding)
            except Exception as e:
                logger.warning(f"Error extracting features for document {i}: {e}")
                # Fallback to just the document text
                embedding = self.embedding_model.encode(doc['content'])
                node_features.append(embedding)
        
        # Convert to tensor
        node_features = torch.stack(node_features).to(next(self.gnn_reranker.parameters()).device)
        
        # Get question embedding
        question_embedding = self.embedding_model.encode(question)
        question_embedding = question_embedding.to(next(self.gnn_reranker.parameters()).device)
        
        # Prepare graph data for GNN
        edge_index, edge_attr = self.prepare_graph_data(doc_graph)
        
        # Apply GNN reranker
        with torch.no_grad():
            self.gnn_reranker.eval()
            node_repr = self.gnn_reranker(node_features, edge_index, edge_attr)
            scores = self.gnn_reranker.compute_scores(node_repr, question_embedding)
        
        # Get reranked indices
        if len(scores) == 0:
            reranked_indices = list(range(len(documents)))
        else:
            reranked_indices = scores.argsort(descending=True).cpu().numpy()
        
        # Cap at top_k if specified
        if top_k is not None:
            reranked_indices = reranked_indices[:min(top_k, len(reranked_indices))]
        
        # Return reranked documents
        reranked_docs = [documents[i] for i in reranked_indices]
        logger.info(f"Reranking complete. Reordered {len(reranked_docs)} documents")
        
        return reranked_docs
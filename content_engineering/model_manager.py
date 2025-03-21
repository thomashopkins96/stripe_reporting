import numpy as np
from typing import Dict, Optional, List, Any
from sentence_transformers import SentenceTransformer

class ModelManager:
    """
    Singleton class to manage embedding models across different modules.
    
    This ensures that all modules using embeddings share the same model instances,
    avoiding duplicate loading and memory usage.
    """
    
    _instance = None
    _models = {}
    
    def __new__(cls):
        """
        Ensure only one instance of ModelManager is created (Singleton pattern).
        """
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """
        Initialize the instance only once.
        """
        if not getattr(self, '_initialized', False):
            self._models = {}
            self._initialized = True
    
    def get_model(self, model_name: str = "mixedbread-ai/mxbai-embed-large-v1") -> SentenceTransformer:
        """
        Get or load a sentence transformer model by name.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            
        Returns:
            Loaded SentenceTransformer model
        """
        if model_name not in self._models:
            print(f"Loading model: {model_name}")
            self._models[model_name] = SentenceTransformer(model_name)
        return self._models[model_name]
    
    def generate_embeddings(self, 
                           texts: List[str], 
                           model_name: str = "mixedbread-ai/mxbai-embed-large-v1", 
                           average: bool = False) -> np.ndarray:
        """
        Generate embeddings for the given texts.
        
        Args:
            texts: List of text chunks to embed
            model_name: Name of the SentenceTransformer model to use
            average: Whether to average the embeddings (for multiple chunks)
            
        Returns:
            Embeddings array or averaged embedding vector if average=True
        """
        if not texts:
            return None
            
        model = self.get_model(model_name)
        embeddings = model.encode(texts)
        
        if average and len(texts) > 1:
            return np.mean(embeddings, axis=0)
        
        return embeddings
    
    def clear_models(self):
        """
        Clear all loaded models to free memory.
        """
        self._models = {}
    
    def list_loaded_models(self) -> List[str]:
        """
        List all currently loaded models.
        
        Returns:
            List of model names that are currently loaded
        """
        return list(self._models.keys())
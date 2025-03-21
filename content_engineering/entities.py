import spacy

class EntityExtractor:
    _instance = None
    _models = {}
    spacy.prefer_gpu()
    
    def __new__(cls):
        """
        Ensure only one instance of ModelManager is created (Singleton pattern).
        """
        if cls._instance is None:
            cls._instance = super(EntityExtractor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """
        Initialize the instance only once.
        """
        if not getattr(self, '_initialized', False):
            self._models = {}
            self._initialized = True
            
    def get_model(self, model_name="en_core_web_lg"):
        """
        Get or load a spaCy model by name.
        
        Args:
            model_name: Name of the spaCy model to use
            
        Returns:
            Loaded spaCy model
        """
        if model_name not in self._models:
            print(f"Loading spaCy model: {model_name}")
            self._models[model_name] = spacy.load(model_name)
        return self._models[model_name]
            
            
    def extract_entities(self, texts, model_name="en_core_web_trf"):
        """
        Extract entities from a list of texts using the specified model.
        
        Args:
            texts: List of text strings to process
            model_name: Name of the spaCy model to use
            
        Returns:
            List of dictionaries containing text and extracted entities
        """
        nlp = self.get_model(model_name)
        result = []
        
        for text in texts:  
            doc = nlp(text)  
            entities = {}  
           
            for ent in doc.ents:  
                try:  
                    entities[ent.text] = ent.label_  # Attempt to add entity to the dictionary  
                except TypeError as e:  
                    print(f"Error: {e} - Entity text: {ent.text}, Entity label: {ent.label_}")  
                    continue  # Skip that entity if there's a TypeError  
            
            result.append({  
                "text": text,  
                "entities": entities  
            })  
        
        return result
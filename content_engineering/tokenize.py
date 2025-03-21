import spacy
from utils.clean import clean_content, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")

def tokenize(text):
    doc = nlp(text)
    return [token.text for token in doc]

def sent_tokenize(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def tokenize_for_bm25(text):
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Apply stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens
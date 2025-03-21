import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_content(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove numbers (optional)
    # text = re.sub(r'\d+', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    cleaned_words = [re.sub(r'[^a-zA-Z0-9\s]', '', word).strip() for word in filtered_text]
    return ' '.join(cleaned_words)
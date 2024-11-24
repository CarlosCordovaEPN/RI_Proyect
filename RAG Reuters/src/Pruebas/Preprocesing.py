import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Tuple

def preprocess_text(text: str, stop_words: set) -> List[str]:
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stop words
        tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        # Remove duplicates while preserving order
        tokens = list(dict.fromkeys(tokens))
        
        return tokens
    
    except Exception as e:
        print(f"Error in text preprocessing: {e}")
        return []

def process_corpus(documents: List[Tuple[str, str]], stop_words: set) -> Tuple[List[Tuple[str, List[str]]], List[str]]:
    processed_documents = []
    dictionary = set()
    
    for filename, content in documents:
        processed_text = preprocess_text(content, stop_words)
        processed_documents.append((filename, processed_text))
        dictionary.update(processed_text)
    
    words_list = sorted(dictionary)
    return processed_documents, words_list
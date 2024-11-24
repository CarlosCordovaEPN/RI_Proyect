import os
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import logging
from typing import Dict

# Configurar logging básico
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentPreprocessor:
    def __init__(self, custom_stopwords_path: str, categories_path: str):
        # Descargar recursos NLTK necesarios
        for resource in ["punkt", "stopwords", "wordnet"]:
            nltk.download(resource, quiet=True)
        
        self.stop_words = self._load_stopwords(custom_stopwords_path)
        self.lemmatizer = WordNetLemmatizer()
        self.categories = self._load_categories(categories_path)
    
    def _load_stopwords(self, file_path: str) -> set:
        with open(file_path, 'r', encoding='utf-8') as f:
            custom_stopwords = set(f.read().splitlines())
        return custom_stopwords.union(set(stopwords.words('english')))
    
    def _load_categories(self, file_path: str) -> Dict:
        categories = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    doc_id = parts[0]
                    doc_categories = parts[1:]
                    categories[doc_id] = doc_categories
        return categories
    
    def preprocess_text(self, text: str) -> str:
        # Limpieza básica
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenización
        tokens = word_tokenize(text)
        
        # Eliminar stopwords y lematizar
        processed_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalpha() and token not in self.stop_words
        ]
        
        return " ".join(processed_tokens)
    
    def process_document(self, file_path: str) -> Dict:
        try:
            # Leer el documento
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Separar título y cuerpo
            parts = content.split("\n", 1)
            title = parts[0] if len(parts) > 0 else ""
            body = parts[1] if len(parts) > 1 else ""
            
            # Procesar título y cuerpo
            processed_title = self.preprocess_text(title)
            processed_body = self.preprocess_text(body)
            
            # Obtener nombre del archivo y categorías
            doc_id = os.path.basename(file_path)
            categories = self.categories.get(doc_id, [])
            
            # Retornar solo los campos necesarios
            return {
                'id': doc_id,
                'title': processed_title,
                'body': processed_body
            }
            
        except Exception as e:
            logger.error(f"Error procesando documento {file_path}: {str(e)}")
            return None
    
    def process_corpus(self, corpus_dir: str) -> pd.DataFrame:

        processed_docs = []
        files = [f for f in os.listdir(corpus_dir) if os.path.isfile(os.path.join(corpus_dir, f))]
        
        for file_name in tqdm(files, desc="Procesando documentos"):
            file_path = os.path.join(corpus_dir, file_name)
            processed_doc = self.process_document(file_path)
            
            if processed_doc:
                processed_docs.append(processed_doc)
        
        df = pd.DataFrame(processed_docs)
        logger.info(f"Procesados {len(df)} documentos exitosamente")
        return df

def main():
    # Configurar rutas
    corpus_dir = ("../data/reuters/test")
    stopwords_file = ("../data/reuters/stopwords.txt")
    categories_file = ("../data/reuters/cats.txt")
    output_file = ("../data/processed/reuters_preprocessed_clean.csv")
    
    # Procesar corpus
    preprocessor = DocumentPreprocessor(stopwords_file, categories_file)
    processed_data = preprocessor.process_corpus(corpus_dir)
    
    # Guardar resultados
    processed_data.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Archivo guardado en: {output_file}")

if __name__ == "__main__":
    main()
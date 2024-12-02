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
        self.normalization_dict = self._load_normalization_dict()
    
    def _load_normalization_dict(self) -> Dict[str, str]:
        """
        Carga el diccionario de normalización con términos comunes en noticias Reuters.
        """
        return {
            # Countries and Regions
            'uk': 'united kingdom',
            'britain': 'united kingdom',
            'great britain': 'united kingdom',
            'united kingdom': 'united kingdom',
            'england': 'united kingdom',
            'british': 'united kingdom',
            'us': 'united states',
            'usa': 'united states',
            'united states of america': 'united states',
            'united states': 'united states',
            'america': 'united states',
            'american': 'united states',
            
            # Organizational Entities
            'un': 'united nations',
            'united nations': 'united nations',
            'who': 'world health organization',
            'world health organization': 'world health organization',
            'imf': 'international monetary fund',
            'international monetary fund': 'international monetary fund',
            'wb': 'world bank',
            'world bank': 'world bank',
            'wto': 'world trade organization',
            'world trade organization': 'world trade organization',
            'nato': 'north atlantic treaty organization',
            'north atlantic treaty organization': 'north atlantic treaty organization',
            
            # Financial Markets
            'nyse': 'new york stock exchange',
            'nasdaq': 'nasdaq stock market',
            'djia': 'dow jones industrial average',
            'sp500': 'standard and poors 500',
            's&p': 'standard and poors 500',
            'ftse': 'financial times stock exchange',
            
            # Central Banks
            'fed': 'federal reserve',
            'federal reserve': 'federal reserve',
            'ecb': 'european central bank',
            'boe': 'bank of england',
            'pboc': 'peoples bank of china',
            'boj': 'bank of japan',
            
            # Economic Indicators
            'gdp': 'gross domestic product',
            'cpi': 'consumer price index',
            'ppi': 'producer price index',
            'pmi': 'purchasing managers index',
            'ism': 'institute supply management',
            'nfp': 'non farm payrolls',
            
            # Business Terms
            'ceo': 'chief executive officer',
            'cfo': 'chief financial officer',
            'coo': 'chief operating officer',
            'ipo': 'initial public offering',
            'ma': 'mergers acquisitions',
            'eps': 'earnings per share',
            'ebit': 'earnings before interest taxes',
            'ebitda': 'earnings before interest taxes depreciation amortization',
            
            # Commodities
            'wti': 'west texas intermediate',
            'brent': 'brent crude oil',
            'opec': 'organization petroleum exporting countries',
            'lng': 'liquefied natural gas',
            
            # Technology
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'iot': 'internet of things',
            'saas': 'software as service',
            'paas': 'platform as service',
            'iaas': 'infrastructure as service',
            
            # Market Terms
            'ytd': 'year to date',
            'yoy': 'year over year',
            'qoq': 'quarter over quarter',
            'mom': 'month over month',
            'ttm': 'trailing twelve months',
            
            # Time Zones
            'est': 'eastern standard time',
            'edt': 'eastern daylight time',
            'gmt': 'greenwich mean time',
            'utc': 'coordinated universal time',
            
            # Common Industry Terms
            'capex': 'capital expenditure',
            'opex': 'operating expense',
            'r&d': 'research development',
            'roi': 'return on investment',
            'roic': 'return on invested capital',
            'roa': 'return on assets',
            'roe': 'return on equity'
        }
    
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
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Tokenización
        tokens = word_tokenize(text)
        
        # Procesamiento de tokens con normalización
        processed_tokens = []
        i = 0
        while i < len(tokens):
            # Intentar normalizar grupos de palabras
            normalized = False
            for window in range(4, 0, -1):
                if i + window <= len(tokens):
                    term = ' '.join(tokens[i:i + window])
                    if term in self.normalization_dict:
                        processed_tokens.append(self.normalization_dict[term])
                        i += window
                        normalized = True
                        break
            
            if not normalized:
                token = tokens[i]
                if token not in self.stop_words and token.isalpha():
                    lemmatized = self.lemmatizer.lemmatize(token)
                    processed_tokens.append(lemmatized)
                i += 1
        
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
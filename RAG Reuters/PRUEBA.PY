import os
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import logging
from typing import Set, List, Dict

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentPreprocessor:
    """
    Clase para el preprocesamiento de documentos de texto.
    
    Attributes:
        stop_words (Set[str]): Conjunto de stopwords combinadas (personalizadas + NLTK)
        lemmatizer: Instancia de WordNetLemmatizer
        categories (Dict[str, str]): Diccionario de categorías por documento
    """
    
    def __init__(self, custom_stopwords_path: str, categories_path: str):
        """
        Inicializa el preprocesador con stopwords y categorías personalizadas.
        
        Args:
            custom_stopwords_path: Ruta al archivo de stopwords personalizadas
            categories_path: Ruta al archivo cats.txt con las categorías
        """
        # Descargar recursos NLTK necesarios
        self._download_nltk_resources()
        
        # Inicializar componentes
        self.stop_words = self._load_stopwords(custom_stopwords_path)
        self.lemmatizer = WordNetLemmatizer()
        self.categories = self._load_categories(categories_path)
        
        logger.info(f"Inicializado con {len(self.stop_words)} stopwords y {len(self.categories)} categorías")
    
    def _download_nltk_resources(self):
        """Descarga los recursos necesarios de NLTK."""
        resources = ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                logger.error(f"Error descargando recurso NLTK {resource}: {str(e)}")
    
    def _load_stopwords(self, file_path: str) -> Set[str]:
        """
        Carga y combina stopwords personalizadas con las de NLTK.
        
        Args:
            file_path: Ruta al archivo de stopwords personalizadas
            
        Returns:
            Set combinado de stopwords
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                custom_stopwords = set(f.read().splitlines())
            all_stopwords = custom_stopwords.union(set(stopwords.words('english')))
            logger.info(f"Cargadas {len(custom_stopwords)} stopwords personalizadas")
            return all_stopwords
        except Exception as e:
            logger.error(f"Error cargando stopwords: {str(e)}")
            return set(stopwords.words('english'))
    
    def _load_categories(self, file_path: str) -> Dict[str, List[str]]:
        """
        Carga las categorías de los documentos desde cats.txt.
        
        Args:
            file_path: Ruta al archivo cats.txt
            
        Returns:
            Diccionario con documento como clave y lista de categorías como valor
        """
        categories = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        doc_id = parts[0]
                        doc_categories = parts[1:]
                        categories[doc_id] = doc_categories
            logger.info(f"Cargadas categorías para {len(categories)} documentos")
            return categories
        except Exception as e:
            logger.error(f"Error cargando categorías: {str(e)}")
            return {}
    
    def preprocess_text(self, text: str, document_stats: Dict = None) -> Dict:
        """
        Preprocesa el texto aplicando varios pasos de limpieza y normalización.
        
        Args:
            text: Texto a preprocesar
            document_stats: Diccionario opcional para guardar estadísticas
            
        Returns:
            Diccionario con el texto procesado y metadatos
        """
        # 1. Extraer contenido relevante (eliminar espacios extra y líneas vacías)
        text = ' '.join(text.split())
        
        # 2. Limpieza de datos
        text = text.lower()  # Normalización a minúsculas
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Eliminar URLs
        text = re.sub(r'@\w+|#\w+', '', text)  # Eliminar menciones y hashtags
        text = re.sub(r'[^\w\s]', ' ', text)  # Eliminar puntuación
        text = re.sub(r'\d+', '', text)  # Eliminar números
        text = re.sub(r'\s+', ' ', text).strip()  # Normalizar espacios
        
        # 3. Tokenización
        tokens = word_tokenize(text)
        
        if document_stats is not None:
            document_stats['original_tokens'] = len(tokens)
        
        # 4. Eliminar stopwords y lematización
        processed_tokens = []
        for token in tokens:
            if token.isalpha() and token not in self.stop_words:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        
        if document_stats is not None:
            document_stats['processed_tokens'] = len(processed_tokens)
            document_stats['unique_tokens'] = len(set(processed_tokens))
        
        # Crear resultado con metadatos
        result = {
            'original_text': text,
            'processed_tokens': processed_tokens,
            'processed_text': ' '.join(processed_tokens),
            'token_count': len(processed_tokens),
            'unique_tokens': len(set(processed_tokens))
        }
        
        return result
    
    def process_document(self, file_path: str) -> Dict:
        """
        Procesa un documento individual y extrae sus características.
        
        Args:
            file_path: Ruta al documento
            
        Returns:
            Diccionario con el contenido procesado y metadatos
        """
        try:
            # Leer el documento
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Separar título y cuerpo
            parts = content.split("\n", 1)
            title = parts[0] if len(parts) > 0 else ""
            body = parts[1] if len(parts) > 1 else ""
            
            # Estadísticas del documento
            doc_stats = {}
            
            # Procesar título y cuerpo
            processed_title = self.preprocess_text(title, doc_stats)
            processed_body = self.preprocess_text(body, doc_stats)
            
            # Obtener categorías si están disponibles
            doc_id = os.path.basename(file_path)
            categories = self.categories.get(doc_id, [])
            
            return {
                'ID': doc_id,
                'title': processed_title,
                'body': processed_body,
                'categories': categories,
                'stats': doc_stats
            }
            
        except Exception as e:
            logger.error(f"Error procesando documento {file_path}: {str(e)}")
            return None
    
    def process_corpus(self, corpus_dir: str) -> pd.DataFrame:
        """
        Procesa todo el corpus de documentos.
        
        Args:
            corpus_dir: Directorio que contiene los documentos
            
        Returns:
            DataFrame con todos los documentos procesados
        """
        processed_docs = []
        files = [f for f in os.listdir(corpus_dir) if os.path.isfile(os.path.join(corpus_dir, f))]
        
        for file_name in tqdm(files, desc="Procesando documentos"):
            file_path = os.path.join(corpus_dir, file_name)
            processed_doc = self.process_document(file_path)
            
            if processed_doc:
                processed_docs.append(processed_doc)
        
        # Crear DataFrame
        df = pd.DataFrame(processed_docs)
        
        # Añadir estadísticas globales
        total_docs = len(df)
        total_tokens = df['body'].apply(lambda x: x['token_count']).sum()
        avg_tokens = total_tokens / total_docs if total_docs > 0 else 0
        
        logger.info(f"""
        Estadísticas del corpus:
        - Total documentos: {total_docs}
        - Total tokens: {total_tokens}
        - Promedio tokens por documento: {avg_tokens:.2f}
        """)
        
        return df

def main():
    # Configurar rutas
    corpus_dir = "data/reuters/test"
    stopwords_file = "data/reuters/stopwords.txt"
    categories_file = "data/reuters/cats.txt"
    output_file = "reuters_preprocessed.csv"
    
    # Crear instancia del preprocesador
    preprocessor = DocumentPreprocessor(stopwords_file, categories_file)
    
    # Procesar el corpus
    logger.info("Iniciando preprocesamiento del corpus...")
    processed_data = preprocessor.process_corpus(corpus_dir)
    
    # Guardar resultados
    processed_data.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Preprocesamiento completado. Resultados guardados en: {output_file}")

if __name__ == "__main__":
    main()
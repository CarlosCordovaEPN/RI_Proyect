import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim.downloader as api
from gensim.utils import simple_preprocess
from typing import Dict
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextVectorizer:
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)
        self.df['id'] = self.df['id'].astype(str)
        self.df['body'] = self.df['body'].astype(str)
        
        self.bow_vectorizer = CountVectorizer(min_df=2, max_df=0.95)
        self.tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.95)
        
        self.vectorization_results = {}
        
        # Cargar modelo pre-entrenado
        logger.info("Cargando modelo Word2Vec pre-entrenado...")
        self.word2vec_model = api.load("word2vec-google-news-300")
        logger.info("Modelo Word2Vec cargado exitosamente")

    def apply_bow(self) -> Dict:
        bow_matrix = self.bow_vectorizer.fit_transform(self.df['body'])
        
        result = {
            'matrix': bow_matrix,
            'vocabulary_size': len(self.bow_vectorizer.vocabulary_),
            'matrix_shape': bow_matrix.shape,
            'sparsity': 1.0 - (bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1])),
            'non_zero_elements': bow_matrix.nnz,
            'memory_usage_mb': bow_matrix.data.nbytes / (1024 * 1024)
        }
        
        self.vectorization_results['BoW'] = result
        return result

    def apply_tfidf(self) -> Dict:
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['body'])
        
        result = {
            'matrix': tfidf_matrix,
            'vocabulary_size': len(self.tfidf_vectorizer.vocabulary_),
            'matrix_shape': tfidf_matrix.shape,
            'sparsity': 1.0 - (tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])),
            'non_zero_elements': tfidf_matrix.nnz,
            'memory_usage_mb': tfidf_matrix.data.nbytes / (1024 * 1024)
        }
        
        self.vectorization_results['TF-IDF'] = result
        return result

    def apply_word2vec(self) -> Dict:
        """
        Aplica Word2Vec usando el modelo pre-entrenado de Google News.
        """
        vector_size = self.word2vec_model.vector_size
        doc_vectors = []
        words_found = 0
        total_words = 0
        
        for doc in self.df['body']:
            words = simple_preprocess(doc)
            total_words += len(words)
            
            vec = np.zeros(vector_size)
            count = 0
            
            for word in words:
                try:
                    if word in self.word2vec_model:
                        vec += self.word2vec_model[word]
                        count += 1
                        words_found += 1
                except KeyError:
                    continue
            
            # Normalizar el vector si encontramos palabras
            if count > 0:
                vec /= count
            
            doc_vectors.append(vec)
        
        doc_vectors_matrix = np.vstack(doc_vectors)
        coverage = (words_found / total_words * 100) if total_words > 0 else 0
        
        result = {
            'matrix': doc_vectors_matrix,
            'vocabulary_size': len(self.word2vec_model),
            'matrix_shape': doc_vectors_matrix.shape,
            'vector_size': vector_size,
            'memory_usage_mb': doc_vectors_matrix.nbytes / (1024 * 1024),
            'non_zero_elements': np.count_nonzero(doc_vectors_matrix),
            'vocabulary_coverage': coverage
        }
        
        self.vectorization_results['Word2Vec'] = result
        return result

    def get_statistics_df(self) -> pd.DataFrame:
        stats_data = []
        
        for method, result in self.vectorization_results.items():
            stats = {
                'Método': method,
                'Tamaño del Vocabulario': result['vocabulary_size'],
                'Documentos': result['matrix_shape'][0],
                'Características': result['matrix_shape'][1],
                'Elementos No Ceros': result['non_zero_elements'],
                'Uso de Memoria (MB)': round(result['memory_usage_mb'], 2)
            }
            
            if method != 'Word2Vec':
                stats['Sparsidad (%)'] = round(result['sparsity'] * 100, 2)
            else:
                stats['Dimensión del Vector'] = result['vector_size']
                stats['Cobertura del Vocabulario (%)'] = round(result['vocabulary_coverage'], 2)
            
            stats_data.append(stats)
        
        return pd.DataFrame(stats_data)

    def save_statistics(self, output_path: str = None):
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f'vectorization_stats_{timestamp}.csv'
        
        stats_df = self.get_statistics_df()
        stats_df.to_csv(output_path, index=False)
        logger.info(f"Estadísticas guardadas en: {output_path}")
        
        return stats_df

def main():
    # Inicializar vectorizador
    vectorizer = TextVectorizer('reuters_preprocessed_clean.csv')
    
    # Aplicar las técnicas de vectorización
    vectorizer.apply_bow()
    vectorizer.apply_tfidf()
    vectorizer.apply_word2vec()
    
    # Guardar estadísticas
    stats_df = vectorizer.save_statistics('vectorization_statistics.csv')
    
    # Mostrar estadísticas en consola
    logger.info("\nEstadísticas de vectorización:")
    print(stats_df.to_string(index=False))

if __name__ == "__main__":
    main()
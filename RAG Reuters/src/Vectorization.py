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
        
        logger.info("Cargando modelo Word2Vec pre-entrenado...")
        self.word2vec_model = api.load("word2vec-google-news-300")
        logger.info("Modelo Word2Vec cargado exitosamente")

    def print_matrix_preview(self, matrix, method_name: str, n_samples: int = 5):
        """Muestra una vista previa de la matriz de vectorización"""
        print(f"\n{'-'*50}\nVista previa de la matriz {method_name}:\n{'-'*50}\n")
        
        if isinstance(matrix, np.ndarray):
            preview = matrix[:n_samples]
        else:  # Para matrices sparse
            preview = matrix[:n_samples].toarray()
            
        print(f"Primeras {n_samples} filas:\n{preview}\n")
        print(f"Forma de la matriz: {matrix.shape}\n")
        
        # Mostrar algunas estadísticas básicas
        if isinstance(matrix, np.ndarray):
            print(f"Media: {np.mean(matrix):.4f}\n")
            print(f"Desviación estándar: {np.std(matrix):.4f}\n")
        else:
            dense_matrix = matrix.toarray()
            print(f"Media: {np.mean(dense_matrix):.4f}\n")
            print(f"Desviación estándar: {np.std(dense_matrix):.4f}\n")
        print(f"Elementos no ceros: {np.count_nonzero(preview)}\n")

    def apply_bow(self) -> Dict:
        logger.info("Aplicando vectorización BoW...\n")
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
        self.print_matrix_preview(bow_matrix, "BoW")
        return result

    def apply_tfidf(self) -> Dict:
        logger.info("Aplicando vectorización TF-IDF...\n")
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
        self.print_matrix_preview(tfidf_matrix, "TF-IDF")
        return result

    def apply_word2vec(self) -> Dict:
        logger.info("Aplicando vectorización Word2Vec...\n")
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
        self.print_matrix_preview(doc_vectors_matrix, "Word2Vec")
        return result

    def get_statistics_df(self) -> pd.DataFrame:
        """
        Genera un DataFrame con las estadísticas de todas las técnicas de vectorización.
        """
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
            
            if method in ['BoW', 'TF-IDF']:
                stats['Sparsidad (%)'] = round(result['sparsity'] * 100, 2)
                stats['Dimensión del Vector'] = None
                stats['Cobertura del Vocabulario (%)'] = None
            else:  # Word2Vec
                stats['Sparsidad (%)'] = None
                stats['Dimensión del Vector'] = result['vector_size']
                stats['Cobertura del Vocabulario (%)'] = round(result['vocabulary_coverage'], 2)
            
            stats_data.append(stats)
        
        return pd.DataFrame(stats_data)

def main():
    # Inicializar vectorizador
    vectorizer = TextVectorizer('reuters_preprocessed_clean.csv')
    
    # Aplicar las técnicas de vectorización
    print("\nAplicando vectorizaciones...\n")
    vectorizer.apply_bow()
    vectorizer.apply_tfidf()
    vectorizer.apply_word2vec()
    
    # Mostrar estadísticas comparativas
    print("\nResumen comparativo de las técnicas de vectorización:\n")
    stats_df = vectorizer.get_statistics_df()
    print(stats_df.to_string(index=False))

if __name__ == "__main__":
    main()

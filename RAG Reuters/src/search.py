import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import logging
from tqdm import tqdm
from Vectorization import TextVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchEngine:
    def __init__(self, data_path: str):
        """Inicializa el motor de búsqueda con el corpus procesado."""
        self.corpus_df = pd.read_csv(data_path)
        logger.info(f"Corpus cargado con {len(self.corpus_df)} documentos")
        
        # Inicializar y aplicar vectorizaciones
        self.vectorizer = TextVectorizer(data_path)
        
        logger.info("Aplicando vectorizaciones...")
        self.bow_results = self.vectorizer.apply_bow()
        self.tfidf_results = self.vectorizer.apply_tfidf()
        self.w2v_results = self.vectorizer.apply_word2vec()
        
        # Guardar matrices
        self.bow_matrix = self.bow_results['matrix']
        self.tfidf_matrix = self.tfidf_results['matrix']
        self.w2v_matrix = self.w2v_results['matrix']
    
    def _search_bow(self, query: str, num_results: int) -> List[Dict]:
        """Búsqueda usando BoW."""
        query_vector = self.vectorizer.bow_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.bow_matrix).flatten()
        return self._format_results(similarities, num_results)
    
    def _search_tfidf(self, query: str, num_results: int) -> List[Dict]:
        """Búsqueda usando TF-IDF."""
        query_vector = self.vectorizer.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        return self._format_results(similarities, num_results)
    
    def _search_word2vec(self, query: str, num_results: int) -> List[Dict]:
        """Búsqueda usando Word2Vec."""
        words = query.lower().split()
        vector_size = self.vectorizer.word2vec_model.vector_size
        query_vector = np.zeros(vector_size)
        words_found = 0
        
        for word in words:
            if word in self.vectorizer.word2vec_model:
                query_vector += self.vectorizer.word2vec_model[word]
                words_found += 1
        
        if words_found == 0:
            return []
        
        query_vector /= words_found
        similarities = cosine_similarity(query_vector.reshape(1, -1), self.w2v_matrix).flatten()
        return self._format_results(similarities, num_results)
    
    def _format_results(self, similarities: np.ndarray, num_results: int) -> List[Dict]:
        """Formatea los resultados de búsqueda."""
        # Obtener índices de documentos con similitud > 0 y ordenarlos
        valid_indices = np.where(similarities > 0)[0]
        top_indices = valid_indices[np.argsort(similarities[valid_indices])[-num_results:][::-1]]
        
        results = []
        for idx in top_indices:
            doc_info = self.corpus_df.iloc[idx]
            results.append({
                'id': doc_info['id'],
                'title': doc_info['title'],
                'preview': doc_info['body'][:200] + '...',  # Primeros 200 caracteres
                'score': float(similarities[idx])
            })
        
        return results
    
    def search(self, query: str, method: str = 'tfidf', num_results: int = 5) -> List[Dict]:
        """
        Realiza la búsqueda de documentos.
        
        Args:
            query: Consulta del usuario
            method: Método de búsqueda ('bow', 'tfidf', o 'word2vec')
            num_results: Número de resultados a retornar
            
        Returns:
            Lista de documentos relevantes con sus scores
        """
        # Preprocesar la consulta
        query = query.lower()
        
        # Realizar búsqueda según el método
        if method == 'bow':
            return self._search_bow(query, num_results)
        elif method == 'tfidf':
            return self._search_tfidf(query, num_results)
        elif method == 'word2vec':
            return self._search_word2vec(query, num_results)
        else:
            raise ValueError(f"Método {method} no soportado")
    
    def print_results(self, results: List[Dict], query: str, method: str):
        """Imprime los resultados de manera formateada."""
        print(f"\nResultados para '{query}' usando {method.upper()}:")
        print("-" * 80)
        
        if not results:
            print("No se encontraron resultados.")
            return
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['id']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Título: {result['title']}")
            print(f"Preview: {result['preview']}")
            print("-" * 80)

def main():
    try:
        # Inicializar motor de búsqueda
        search_engine = SearchEngine('../data/processed/reuters_preprocessed_clean.csv')
        
        # Bucle interactivo para búsquedas
        while True:
            # Obtener consulta del usuario
            query = input("\nIngrese su consulta (o 'salir' para terminar): ")
            if query.lower() == 'salir':
                break
            
            # Obtener método de búsqueda
            print("\nMétodos disponibles:")
            print("1. BoW")
            print("2. TF-IDF")
            print("3. Word2Vec")
            method_choice = input("Elija el método (1-3): ")
            
            # Mapear elección a método
            method_map = {'1': 'bow', '2': 'tfidf', '3': 'word2vec'}
            method = method_map.get(method_choice, 'tfidf')
            
            # Realizar búsqueda
            results = search_engine.search(query, method=method)
            search_engine.print_results(results, query, method)
            
    except KeyboardInterrupt:
        print("\nBúsqueda terminada.")
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
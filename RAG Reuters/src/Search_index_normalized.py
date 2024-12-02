import pandas as pd
import numpy as np
from typing import List, Dict
import logging
import math
import csv
from collections import defaultdict
from Read_processing_normalized import DocumentPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchEngine:
    def __init__(self, corpus_path: str, index_path: str, stopwords_path: str, categories_path: str):
        """
        Inicializa el motor de búsqueda cargando el corpus y el índice guardado.
        """
        # Inicializar el preprocessor para normalización
        self.preprocessor = DocumentPreprocessor(stopwords_path, categories_path)
        
        # Cargar corpus
        self.corpus_df = pd.read_csv(corpus_path)
        self.corpus_df['id'] = self.corpus_df['id'].astype(str)
        self.N = len(self.corpus_df)
        logger.info(f"Corpus cargado con {self.N} documentos")
        
        # Cargar índice invertido
        self.index = self._load_index(index_path)
        logger.info(f"Índice invertido cargado con {len(self.index)} términos")
        
        # Crear mapeo de documentos para acceso rápido
        self.doc_mapping = {str(doc_id): idx for idx, doc_id in enumerate(self.corpus_df['id'])}
        
        # Calcular IDF y vectores una sola vez
        self.idf = self._calculate_idf()
        self.doc_vectors = self._calculate_doc_vectors()
    
    def _load_index(self, index_path: str) -> Dict:
        """Carga el índice invertido desde el archivo CSV."""
        index = defaultdict(lambda: defaultdict(int))
        
        with open(index_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Saltar encabezados
            
            for row in reader:
                if len(row) >= 3:
                    term = row[0]
                    doc_ids = row[1].split()
                    frequencies = [int(f) for f in row[2].split()]
                    
                    for doc_id, freq in zip(doc_ids, frequencies):
                        index[term][doc_id] = freq
        
        return index
    
    def _calculate_idf(self) -> Dict[str, float]:
        """Calcula IDF usando el índice invertido cargado."""
        idf = {}
        for term, doc_freq in self.index.items():
            idf[term] = math.log(self.N / (1 + len(doc_freq)))
        return idf
    
    def _calculate_doc_vectors(self) -> Dict[str, Dict[str, float]]:
        """Calcula vectores TF-IDF usando el índice invertido."""
        doc_vectors = defaultdict(dict)
        
        for term, doc_freqs in self.index.items():
            for doc_id, tf in doc_freqs.items():
                tf_idf = (1 + math.log(tf)) * self.idf[term]
                doc_vectors[doc_id][term] = tf_idf
        
        return doc_vectors
    
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """Realiza la búsqueda usando TF-IDF con normalización."""
        # Normalizar la consulta
        normalized_query = self.preprocessor.preprocess_text(query)
        logger.info(f"Query original: '{query}'")
        logger.info(f"Query normalizada: '{normalized_query}'")
        
        # Procesar la consulta normalizada
        query_terms = normalized_query.split()
        query_vector = {}
        
        # Calcular TF para la consulta
        term_freq = {}
        for term in query_terms:
            term_freq[term] = term_freq.get(term, 0) + 1
        
        # Calcular TF-IDF para la consulta
        for term, tf in term_freq.items():
            if term in self.idf:
                query_vector[term] = (1 + math.log(tf)) * self.idf[term]
        
        # Calcular similitudes
        similarities = []
        for doc_id, doc_vector in self.doc_vectors.items():
            similarity = self._cosine_similarity(query_vector, doc_vector)
            if similarity > 0:
                similarities.append((doc_id, similarity))
        
        # Ordenar por similitud
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return self._format_results(similarities[:num_results])
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calcula similitud coseno entre dos vectores dispersos."""
        common_terms = set(vec1.keys()) & set(vec2.keys())
        
        if not common_terms:
            return 0.0
        
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        norm1 = math.sqrt(sum(v*v for v in vec1.values()))
        norm2 = math.sqrt(sum(v*v for v in vec2.values()))
        
        return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
    
    def _format_results(self, similarity_pairs: List[tuple]) -> List[Dict]:
        """Formatea los resultados de búsqueda."""
        results = []
        
        for doc_id, score in similarity_pairs:
            doc_mask = self.corpus_df['id'] == doc_id
            if any(doc_mask):
                doc_info = self.corpus_df[doc_mask].iloc[0]
                
                result = {
                    'doc_id': doc_id,
                    'score': float(score)
                }
                
                if 'title' in doc_info:
                    result['title'] = doc_info['title']
                if 'body' in doc_info:
                    result['preview'] = str(doc_info['body'])[:200] + '...'
                
                results.append(result)
        
        return results
    
    def print_results(self, results: List[Dict], query: str):
        """Imprime los resultados de manera formateada."""
        print(f"\nResultados para '{query}':")
        print("-" * 80)
        
        if not results:
            print("No se encontraron resultados.")
            return
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Documento: {result['doc_id']}")
            print(f"Score: {result['score']:.4f}")
            
            if 'title' in result:
                print(f"Título: {result['title']}")
            
            if 'preview' in result:
                print(f"Preview: {result['preview']}")
            
            print("-" * 80)

def main():
    try:
        # Inicializar motor de búsqueda
        search_engine = SearchEngine(
            corpus_path='../data/processed/reuters_preprocessed_clean.csv',
            index_path='../data/processed/inverted_index.csv',
            stopwords_path='../data/reuters/stopwords.txt',
            categories_path='../data/reuters/cats.txt'
        )
        
        while True:
            query = input("\nIngrese su consulta (o 'salir' para terminar): ")
            if query.lower() == 'salir':
                break
            
            results = search_engine.search(query)
            search_engine.print_results(results, query)
            
    except KeyboardInterrupt:
        print("\nBúsqueda terminada.")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
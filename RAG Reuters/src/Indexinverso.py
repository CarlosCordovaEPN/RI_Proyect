import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
import math
import csv
from collections import defaultdict
from tqdm import tqdm
import time
from dataclasses import dataclass
from pathlib import Path

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Clase para almacenar un resultado de búsqueda."""
    doc_id: str
    score: float
    title: str
    preview: str
    rank: int = 0

class SearchEngine:
    def __init__(self, corpus_path: str, index_path: str):
        """
        Inicializa el motor de búsqueda.
        
        Args:
            corpus_path: Ruta al archivo CSV del corpus
            index_path: Ruta al archivo CSV del índice invertido
        """
        self._validate_files(corpus_path, index_path)
        
        start_time = time.time()
        self._load_corpus(corpus_path)
        self._load_index(index_path)
        self._initialize_vectors()
        
        logger.info(f"Motor de búsqueda inicializado en {time.time() - start_time:.2f} segundos")
        self._print_statistics()

    def _validate_files(self, corpus_path: str, index_path: str):
        """Valida la existencia de los archivos necesarios."""
        for path in [corpus_path, index_path]:
            if not Path(path).is_file():
                raise FileNotFoundError(f"No se encontró el archivo: {path}")

    def _load_corpus(self, corpus_path: str):
        """Carga y preprocesa el corpus."""
        logger.info("Cargando corpus...")
        self.corpus_df = pd.read_csv(corpus_path)
        self.corpus_df['id'] = self.corpus_df['id'].astype(str)
        self.N = len(self.corpus_df)
        
        # Crear mapeo de documentos para acceso rápido
        self.doc_mapping = {str(doc_id): idx for idx, doc_id in enumerate(self.corpus_df['id'])}

    def _load_index(self, index_path: str):
        """Carga el índice invertido desde CSV."""
        logger.info("Cargando índice invertido...")
        self.index = defaultdict(lambda: defaultdict(int))
        
        with open(index_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Saltar encabezados
            
            for row in tqdm(reader, desc="Cargando índice"):
                if len(row) >= 3:
                    term = row[0]
                    doc_ids = row[1].split()
                    frequencies = [int(f) for f in row[2].split()]
                    
                    for doc_id, freq in zip(doc_ids, frequencies):
                        self.index[term][doc_id] = freq

    def _initialize_vectors(self):
        """Inicializa los vectores TF-IDF."""
        logger.info("Calculando vectores TF-IDF...")
        
        # Calcular IDF
        self.idf = {
            term: math.log(self.N / (1 + len(doc_freq)))
            for term, doc_freq in self.index.items()
        }
        
        # Calcular vectores de documentos
        self.doc_vectors = defaultdict(dict)
        for term, doc_freqs in tqdm(self.index.items(), desc="Calculando vectores"):
            for doc_id, tf in doc_freqs.items():
                self.doc_vectors[doc_id][term] = (1 + math.log(tf)) * self.idf[term]

    def _print_statistics(self):
        """Imprime estadísticas del motor de búsqueda."""
        logger.info("-" * 50)
        logger.info("Estadísticas del motor de búsqueda:")
        logger.info(f"- Documentos en el corpus: {self.N:,}")
        logger.info(f"- Términos únicos en el índice: {len(self.index):,}")
        logger.info(f"- Tamaño promedio del vocabulario por documento: "
                   f"{sum(len(v) for v in self.doc_vectors.values()) / self.N:.1f}")
        logger.info("-" * 50)

    def _process_query(self, query: str) -> Dict[str, float]:
        """Procesa y vectoriza la consulta."""
        terms = query.lower().split()
        term_freq = {}
        
        # Calcular TF
        for term in terms:
            term_freq[term] = term_freq.get(term, 0) + 1
        
        # Calcular TF-IDF
        query_vector = {}
        for term, tf in term_freq.items():
            if term in self.idf:
                query_vector[term] = (1 + math.log(tf)) * self.idf[term]
        
        return query_vector

    def _calculate_similarity(self, query_vector: Dict[str, float], doc_vector: Dict[str, float]) -> float:
        """Calcula la similitud coseno entre la consulta y un documento."""
        common_terms = set(query_vector.keys()) & set(doc_vector.keys())
        
        if not common_terms:
            return 0.0
        
        dot_product = sum(query_vector[term] * doc_vector[term] for term in common_terms)
        query_norm = math.sqrt(sum(v*v for v in query_vector.values()))
        doc_norm = math.sqrt(sum(v*v for v in doc_vector.values()))
        
        return dot_product / (query_norm * doc_norm) if query_norm > 0 and doc_norm > 0 else 0.0

    def search(self, query: str, num_results: int = 5, min_score: float = 0.0) -> List[SearchResult]:
        """
        Realiza la búsqueda en el corpus.
        
        Args:
            query: Consulta del usuario
            num_results: Número máximo de resultados
            min_score: Score mínimo para considerar un resultado
            
        Returns:
            Lista de resultados ordenados por relevancia
        """
        start_time = time.time()
        query_vector = self._process_query(query)
        
        if not query_vector:
            logger.warning("La consulta no contiene términos válidos del corpus")
            return []
        
        # Calcular similitudes
        similarities = [
            (doc_id, self._calculate_similarity(query_vector, doc_vector))
            for doc_id, doc_vector in self.doc_vectors.items()
        ]
        
        # Filtrar y ordenar resultados
        valid_results = [
            (doc_id, score) for doc_id, score in similarities
            if score > min_score
        ]
        valid_results.sort(key=lambda x: x[1], reverse=True)
        
        # Formatear resultados
        results = []
        for rank, (doc_id, score) in enumerate(valid_results[:num_results], 1):
            doc_info = self.corpus_df[self.corpus_df['id'] == doc_id].iloc[0]
            
            result = SearchResult(
                doc_id=doc_id,
                score=score,
                title=doc_info['title'],
                preview=f"{str(doc_info['body'])[:200]}...",
                rank=rank
            )
            results.append(result)
        
        search_time = time.time() - start_time
        logger.info(f"Búsqueda completada en {search_time:.3f} segundos")
        logger.info(f"Encontrados {len(results)} resultados de {len(valid_results)} documentos relevantes")
        
        return results

    def print_results(self, results: List[SearchResult], query: str):
        """Imprime los resultados de manera formateada."""
        print(f"\nResultados para: '{query}'")
        print("=" * 80)
        
        if not results:
            print("No se encontraron resultados.")
            return
        
        for result in results:
            print(f"\nRank {result.rank}")
            print(f"Documento: {result.doc_id}")
            print(f"Score: {result.score:.4f}")
            print(f"Título: {result.title}")
            print(f"Preview: {result.preview}")
            print("-" * 80)

def main():
    try:
        # Inicializar motor de búsqueda
        search_engine = SearchEngine(
            corpus_path='../data/processed/reuters_preprocessed_clean.csv',
            index_path='../data/processed/inverted_index.csv'
        )
        
        # Bucle de búsqueda interactiva
        while True:
            print("\n" + "=" * 80)
            query = input("Ingrese su consulta (o 'salir' para terminar): ")
            
            if query.lower() == 'salir':
                break
            
            if not query.strip():
                print("Por favor ingrese una consulta válida.")
                continue
            
            try:
                results = search_engine.search(query)
                search_engine.print_results(results, query)
            except Exception as e:
                logger.error(f"Error al procesar la consulta: {str(e)}")
                print("Ocurrió un error al procesar su consulta. Por favor intente de nuevo.")
        
        print("\n¡Gracias por usar el motor de búsqueda!")
        
    except KeyboardInterrupt:
        print("\n\nBúsqueda terminada por el usuario.")
    except Exception as e:
        logger.error(f"Error fatal: {str(e)}")
        raise

if __name__ == "__main__":
    main()
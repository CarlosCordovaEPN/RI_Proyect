import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from tqdm import tqdm
import csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)
        self.doc_lengths = {}
        self.term_frequencies = defaultdict(lambda: defaultdict(int))
        
    def build_index(self, df: pd.DataFrame) -> None:
        logger.info(f"Iniciando construcción del índice para {len(df)} documentos...")
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Construyendo índice"):
            doc_id = str(row['id'])  # Convertir a string
            text = str(row['body'])  # Convertir a string
            
            terms = text.split()
            self.doc_lengths[doc_id] = len(terms)
            
            # Procesar términos únicos
            term_set = set(terms)
            for term in term_set:
                self.index[term].append(doc_id)
                self.term_frequencies[term][doc_id] = terms.count(term)
    
    def get_posting_list(self, term: str) -> list:
        return self.index.get(term, [])
    
    def get_term_frequency(self, term: str, doc_id: str) -> int:
        return self.term_frequencies[term][doc_id]
    
    def save_index_to_csv(self, output_path: str = 'inverted_index.csv'):
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['term', 'doc_ids', 'frequencies'])
            
            for term in self.index:
                doc_ids = self.index[term]
                # Convertir todas las frecuencias a string
                frequencies = [str(self.term_frequencies[term][doc_id]) for doc_id in doc_ids]
                writer.writerow([
                    str(term),
                    ' '.join(str(doc_id) for doc_id in doc_ids),  # Unir con espacios
                    ' '.join(frequencies)  # Unir con espacios
                ])
        
        logger.info(f"Índice guardado en: {output_path}")
    
    def get_statistics(self) -> dict:
        total_terms = len(self.index)
        total_docs = len(self.doc_lengths)
        
        if total_terms > 0:
            avg_posting_length = sum(len(posting) for posting in self.index.values()) / total_terms
            most_frequent_term = max(self.index.keys(), key=lambda t: len(self.index[t]))
        else:
            avg_posting_length = 0
            most_frequent_term = "N/A"
            
        if total_docs > 0:
            longest_doc = max(self.doc_lengths.items(), key=lambda x: x[1])[0]
            avg_doc_length = sum(self.doc_lengths.values()) / total_docs
        else:
            longest_doc = "N/A"
            avg_doc_length = 0
        
        stats = {
            "Número total de términos únicos": str(total_terms),
            "Número total de documentos": str(total_docs),
            "Promedio de documentos por término": str(round(avg_posting_length, 2)),
            "Término más frecuente": str(most_frequent_term),
            "Documento más largo": str(longest_doc),
            "Longitud promedio de documento": str(round(avg_doc_length, 2))
        }
        
        return stats
    
    def save_statistics(self, output_path: str = 'index_statistics.csv'):
        stats = self.get_statistics()
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Métrica', 'Valor'])
            for metric, value in stats.items():
                writer.writerow([str(metric), str(value)])  # Convertir ambos a string
        
        logger.info(f"Estadísticas guardadas en: {output_path}")

def main():
    try:
        # Cargar datos preprocesados
        df = pd.read_csv('reuters_preprocessed_clean.csv')
        
        # Construir índice
        index = InvertedIndex()
        index.build_index(df)
        
        # Guardar índice y estadísticas
        index.save_index_to_csv()
        index.save_statistics()
        
        # Mostrar estadísticas
        stats = index.get_statistics()
        logger.info("\nEstadísticas del índice invertido:")
        for metric, value in stats.items():
            logger.info(f"{metric}: {value}")
        
        # Ejemplo de uso
        if len(index.index) > 0:
            example_term = list(index.index.keys())[0]
            posting_list = index.get_posting_list(example_term)
            logger.info(f"\nEjemplo de posting list para el término '{example_term}':")
            logger.info(f"Documentos que contienen el término: {posting_list[:5]}...")
    
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}")

if __name__ == "__main__":
    main()
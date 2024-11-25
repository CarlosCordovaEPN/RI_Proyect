import pandas as pd
import numpy as np
from typing import List, Dict
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_queries(corpus_path: str, output_path: str = 'test_queries.csv'):
    """
    Genera consultas de prueba basadas en el corpus Reuters.
    """
    # Cargar corpus
    logger.info("Cargando corpus...")
    df = pd.read_csv(corpus_path)
    
    # Definir consultas de prueba relevantes para Reuters
    test_queries = [
        {
            'query': 'grain wheat corn',
            'keywords': ['grain', 'wheat', 'corn', 'agriculture', 'harvest', 'farm']
        },
        {
            'query': 'crude oil prices',
            'keywords': ['crude', 'oil', 'petroleum', 'prices', 'barrel']
        },
        {
            'query': 'trade deficit',
            'keywords': ['trade', 'deficit', 'export', 'import', 'balance']
        },
        {
            'query': 'interest rates bank',
            'keywords': ['interest', 'rates', 'bank', 'fed', 'federal', 'reserve']
        },
        {
            'query': 'stock market exchange',
            'keywords': ['stock', 'market', 'exchange', 'shares', 'trading']
        },
        {
            'query': 'gold silver mining',
            'keywords': ['gold', 'silver', 'mining', 'metals', 'precious']
        },
        {
            'query': 'coffee production export',
            'keywords': ['coffee', 'production', 'export', 'bean', 'crop']
        },
        {
            'query': 'european currency markets',
            'keywords': ['european', 'currency', 'markets', 'exchange', 'ecu']
        },
        {
            'query': 'japan manufacturing trade',
            'keywords': ['japan', 'manufacturing', 'trade', 'export', 'industry']
        },
        {
            'query': 'corporate earnings profit',
            'keywords': ['corporate', 'earnings', 'profit', 'revenue', 'quarter']
        }
    ]
    
    # Encontrar documentos relevantes para cada consulta
    queries_data = []
    
    for query_info in test_queries:
        query = query_info['query']
        keywords = query_info['keywords']
        
        # Buscar documentos que contengan las palabras clave
        relevant_docs = []
        
        for _, row in df.iterrows():
            text = str(row['body']).lower()
            # Un documento es relevante si contiene al menos 3 palabras clave
            if sum(1 for keyword in keywords if keyword in text) >= 3:
                relevant_docs.append(str(row['id']))
        
        # Limitar a máximo 10 documentos relevantes por consulta
        relevant_docs = relevant_docs[:10]
        
        if relevant_docs:
            queries_data.append({
                'query': query,
                'relevant_docs': ' '.join(relevant_docs)
            })
    
    # Crear DataFrame y guardar
    queries_df = pd.DataFrame(queries_data)
    queries_df.to_csv(output_path, index=False)
    
    logger.info(f"Archivo de consultas de prueba guardado en: {output_path}")
    logger.info(f"Total de consultas generadas: {len(queries_df)}")
    
    # Mostrar ejemplo de las consultas generadas
    print("\nEjemplo de consultas generadas:")
    print(queries_df.to_string())

def main():
    try:
        # Generar consultas de prueba
        generate_test_queries(
            corpus_path='../data/processed/reuters_preprocessed_clean.csv',
            output_path='../data/processed/test_queries.csv'
        )
        
    except Exception as e:
        logger.error(f"Error durante la generación de consultas: {str(e)}")
        raise

if __name__ == "__main__":
    main()
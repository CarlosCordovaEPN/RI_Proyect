import os
import logging
from pathlib import Path
import argparse

# Importar los módulos del sistema
from Read_processing import DocumentPreprocessor
from Vectorization import TextVectorizer
from search_index import SearchEngine
from evaluation import SystemEvaluator

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SearchSystem:
    def __init__(self, config):
        self.config = config
        self.setup_directories()
        
    def setup_directories(self):
        """Crea las estructuras de directorios necesarias."""
        for path in [
            self.config['processed_dir'],
            self.config['evaluation_dir'],
            self.config['results_dir']
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)
            
    def process_documents(self):
        """Procesa los documentos del corpus."""
        logger.info("Iniciando procesamiento de documentos...")
        
        preprocessor = DocumentPreprocessor(
            custom_stopwords_path=self.config['stopwords_path'],
            categories_path=self.config['categories_path']
        )
        
        processed_data = preprocessor.process_corpus(self.config['corpus_dir'])
        processed_path = os.path.join(self.config['processed_dir'], 'reuters_preprocessed_clean.csv')
        processed_data.to_csv(processed_path, index=False, encoding='utf-8')
        
        logger.info(f"Documentos procesados guardados en: {processed_path}")
        return processed_path
    
    def create_vectors(self, processed_path):
        """Crea las representaciones vectoriales de los documentos."""
        logger.info("Generando vectorizaciones...")
        
        vectorizer = TextVectorizer(processed_path)
        vectorizer.apply_bow()
        vectorizer.apply_tfidf()
        vectorizer.apply_word2vec()
        
        # Guardar estadísticas
        stats_path = os.path.join(self.config['results_dir'], 'vectorization_stats.csv')
        vectorizer.save_statistics(stats_path)
        
        logger.info(f"Estadísticas de vectorización guardadas en: {stats_path}")
        return stats_path
    
    def initialize_search_engine(self, processed_path):
        """Inicializa el motor de búsqueda."""
        logger.info("Inicializando motor de búsqueda...")
        
        self.search_engine = SearchEngine(
            corpus_path=processed_path,
            index_path=self.config['index_path']
        )
        
        return self.search_engine
    
    def evaluate_system(self, search_engine):
        """Evalúa el rendimiento del sistema."""
        logger.info("Iniciando evaluación del sistema...")
        
        evaluator = SystemEvaluator(
            search_engine=search_engine,
            test_queries_path=self.config['test_queries_path']
        )
        
        # Evaluar para diferentes valores de k
        results = evaluator.evaluate_system(k_values=[5, 10, 20])
        
        # Generar visualizaciones y guardar resultados
        evaluator.plot_results(self.config['evaluation_dir'])
        evaluator.save_results(self.config['evaluation_dir'])
        
        # Mostrar resumen
        evaluator.print_summary()
        
        return results
    
    def interactive_search(self):
        """Inicia el modo de búsqueda interactiva."""
        logger.info("Iniciando modo de búsqueda interactiva...")
        
        while True:
            print("\n" + "=" * 80)
            query = input("Ingrese su consulta (o 'salir' para terminar): ")
            
            if query.lower() == 'salir':
                break
                
            if not query.strip():
                print("Por favor ingrese una consulta válida.")
                continue
            
            try:
                results = self.search_engine.search(query)
                self.search_engine.print_results(results, query)
            except Exception as e:
                logger.error(f"Error al procesar la consulta: {str(e)}")
                print("Ocurrió un error al procesar su consulta. Por favor intente de nuevo.")

def main():
    # Configuración del parser de argumentos
    parser = argparse.ArgumentParser(description='Sistema de Búsqueda de Documentos')
    parser.add_argument('--mode', choices=['process', 'search', 'evaluate', 'full'],
                      default='search', help='Modo de operación del sistema')
    args = parser.parse_args()
    
    # Configuración del sistema
    config = {
        'corpus_dir': '../data/reuters/test',
        'processed_dir': '../data/processed',
        'evaluation_dir': '../evaluation',
        'results_dir': '../results',
        'stopwords_path': '../data/reuters/stopwords.txt',
        'categories_path': '../data/reuters/cats.txt',
        'index_path': '../data/processed/inverted_index.csv',
        'test_queries_path': '../data/processed/test_queries.csv'
    }
    
    try:
        system = SearchSystem(config)
        processed_path = None
        
        if args.mode in ['process', 'full']:
            processed_path = system.process_documents()
            system.create_vectors(processed_path)
        
        if args.mode in ['search', 'evaluate', 'full']:
            if not processed_path:
                processed_path = os.path.join(config['processed_dir'], 'reuters_preprocessed_clean.csv')
            
            search_engine = system.initialize_search_engine(processed_path)
            
            if args.mode in ['evaluate', 'full']:
                system.evaluate_system(search_engine)
            
            if args.mode in ['search', 'full']:
                system.interactive_search()
        
    except KeyboardInterrupt:
        print("\nPrograma terminado por el usuario.")
    except Exception as e:
        logger.error(f"Error fatal: {str(e)}")
        raise

if __name__ == "__main__":
    main()
import os
import sys
import logging

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from flask import Flask, render_template, request, jsonify
from src.mainsearchsystem import SearchSystem
from src.Vectorization import TextVectorizer
import pandas as pd

app = Flask(__name__)

# Configuración del sistema
config = {
    'corpus_dir': os.path.join(current_dir, 'data/reuters/test'),
    'processed_dir': os.path.join(current_dir, 'data/processed'),
    'evaluation_dir': os.path.join(current_dir, 'evaluation'),
    'results_dir': os.path.join(current_dir, 'results'),
    'stopwords_path': os.path.join(current_dir, 'data/reuters/stopwords.txt'),
    'categories_path': os.path.join(current_dir, 'data/reuters/cats.txt'),
    'index_path': os.path.join(current_dir, 'data/processed/inverted_index.csv'),
    'test_queries_path': os.path.join(current_dir, 'data/processed/test_queries.csv')
}

# Inicializar el vectorizador y calcular estadísticas
processed_path = os.path.join(current_dir, 'data/processed/reuters_preprocessed_clean.csv')
logger.info(f"Ruta del archivo procesado: {processed_path}")

vectorizer = None
vectorization_stats = {}

def initialize_vectorization_stats():
    global vectorizer, vectorization_stats
    try:
        logger.debug("Iniciando cálculo de estadísticas de vectorización...")
        
        # Inicializar el vectorizador
        vectorizer = TextVectorizer(processed_path)
        
        # Aplicar las técnicas de vectorización
        logger.debug("Aplicando BoW...")
        bow_stats = vectorizer.apply_bow()
        logger.debug(f"BoW completado: {bow_stats}")
        
        logger.debug("Aplicando TF-IDF...")
        tfidf_stats = vectorizer.apply_tfidf()
        logger.debug(f"TF-IDF completado: {tfidf_stats}")
        
        logger.debug("Aplicando Word2Vec...")
        word2vec_stats = vectorizer.apply_word2vec()
        logger.debug(f"Word2Vec completado: {word2vec_stats}")
        
        # Obtener el DataFrame de estadísticas
        stats_df = vectorizer.get_statistics_df()
        logger.debug(f"DataFrame de estadísticas generado: {stats_df}")
        
        # Convertir el DataFrame a un formato más adecuado para JSON
        vectorization_stats = stats_df.to_dict('records')
        logger.info("Estadísticas de vectorización inicializadas correctamente")
        logger.debug(f"Estadísticas finales: {vectorization_stats}")
        
    except Exception as e:
        logger.error(f"Error al inicializar estadísticas de vectorización: {str(e)}")
        logger.exception("Detalles del error:")
        vectorization_stats = []

def read_original_reuters(doc_id):
    """Lee el contenido original de un documento Reuters."""
    try:
        file_path = os.path.join(config['corpus_dir'], str(doc_id))
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        logger.error(f"Error leyendo documento {doc_id}: {str(e)}")
    return None

# Inicializar el sistema de búsqueda
search_system = SearchSystem(config)
search_engine = search_system.initialize_search_engine(processed_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        results = search_engine.search(query, num_results=10)
        
        enriched_results = []
        for result in results:
            doc_id = result['doc_id']
            original_text = read_original_reuters(doc_id)
            
            if original_text:
                enriched_results.append({
                    'id': doc_id,
                    'content': original_text,
                    'score': result['score']
                })
        
        return jsonify({
            'status': 'success',
            'results': enriched_results,
            'total': len(enriched_results)
        })
        
    except Exception as e:
        logger.error(f"Error en la búsqueda: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/vectorization-stats')
def get_vectorization_stats():
    """Endpoint para obtener las estadísticas de vectorización"""
    try:
        logger.debug(f"Obteniendo estadísticas de vectorización: {vectorization_stats}")
        
        if not vectorization_stats:
            logger.warning("No hay estadísticas de vectorización disponibles")
            # Si no hay estadísticas, intentamos inicializarlas de nuevo
            initialize_vectorization_stats()
            
            if not vectorization_stats:
                return jsonify({
                    'status': 'error',
                    'message': 'No hay estadísticas disponibles'
                }), 404
        
        return jsonify({
            'status': 'success',
            'stats': vectorization_stats
        })
    except Exception as e:
        logger.error(f"Error al obtener estadísticas de vectorización: {str(e)}")
        logger.exception("Detalles del error:")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    logger.info("Iniciando aplicación...")
    initialize_vectorization_stats()  # Inicializar estadísticas antes de iniciar la app
    app.run(debug=True)
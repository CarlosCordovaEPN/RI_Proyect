import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from flask import Flask, render_template, request, jsonify
from src.mainsearchsystem import SearchSystem
import logging
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

def read_original_reuters(doc_id):
    """Lee el contenido original de un documento Reuters."""
    try:
        file_path = os.path.join(config['corpus_dir'], str(doc_id))
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        logging.error(f"Error leyendo documento {doc_id}: {str(e)}")
    return None

# Inicializar el sistema de búsqueda
search_system = SearchSystem(config)
processed_path = os.path.join(current_dir, 'data/processed/reuters_preprocessed_clean.csv')
search_engine = search_system.initialize_search_engine(processed_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        # Realizar la búsqueda usando tu implementación
        results = search_engine.search(query, num_results=10)
        
        # Enriquecer resultados con el texto original
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
        logging.error(f"Error en la búsqueda: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
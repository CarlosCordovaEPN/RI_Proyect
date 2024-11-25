import pandas as pd
import numpy as np
from typing import List, Dict
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemEvaluator:
    def __init__(self, search_engine, test_queries_path: str):
        self.search_engine = search_engine
        self.test_queries = pd.read_csv(test_queries_path)
        self.evaluation_results = {}
        logger.info(f"Cargadas {len(self.test_queries)} consultas de prueba")
    
    def evaluate_query(self, query: str, relevant_docs: str, k: int = 10) -> Dict:
        """Evalúa una consulta individual."""
        # Convertir relevant_docs a set
        relevant_set = set(relevant_docs.split())
        
        # Realizar búsqueda
        search_results = self.search_engine.search(query, num_results=k)
        retrieved_set = {str(result['doc_id']) for result in search_results}  # Cambiado para coincidir con el formato
        
        # Calcular intersección
        relevant_retrieved = retrieved_set & relevant_set
        
        # Calcular métricas básicas
        precision = len(relevant_retrieved) / len(retrieved_set) if retrieved_set else 0
        recall = len(relevant_retrieved) / len(relevant_set) if relevant_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calcular precisión promedio (AP)
        ap = self._calculate_ap(search_results, relevant_set)
        
        # Calcular DCG y nDCG
        dcg = self._calculate_dcg(search_results, relevant_set, k)
        idcg = self._calculate_idcg(len(relevant_set), k)
        ndcg = dcg / idcg if idcg > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'average_precision': ap,
            'ndcg': ndcg,
            'retrieved_count': len(retrieved_set),
            'relevant_count': len(relevant_set),
            'relevant_retrieved_count': len(relevant_retrieved)
        }
    
    def _calculate_ap(self, results: List, relevant_docs: set) -> float:
        """Calcula Average Precision."""
        precision_sum = 0
        relevant_found = 0
        
        for i, result in enumerate(results, 1):
            if str(result['doc_id']) in relevant_docs:  # Cambiado para coincidir con el formato
                relevant_found += 1
                precision_at_k = relevant_found / i
                precision_sum += precision_at_k
        
        return precision_sum / len(relevant_docs) if relevant_docs else 0
    
    def _calculate_dcg(self, results: List, relevant_docs: set, k: int) -> float:
        """Calcula Discounted Cumulative Gain."""
        dcg = 0
        for i, result in enumerate(results[:k], 1):
            rel = 1 if str(result['doc_id']) in relevant_docs else 0  # Cambiado para coincidir con el formato
            dcg += rel / np.log2(i + 1)
        return dcg
    
    def _calculate_idcg(self, num_relevant: int, k: int) -> float:
        """Calcula Ideal DCG."""
        idcg = 0
        for i in range(min(num_relevant, k)):
            idcg += 1 / np.log2(i + 2)
        return idcg

    def evaluate_system(self, k_values: List[int] = [5, 10, 20]) -> Dict:
        """Evalúa el sistema completo para diferentes valores de k."""
        results = {}
        
        for k in k_values:
            logger.info(f"Evaluando sistema con k={k}")
            query_results = []
            
            for _, row in tqdm(self.test_queries.iterrows(), 
                             total=len(self.test_queries),
                             desc=f"Evaluando consultas (k={k})"):
                
                eval_result = self.evaluate_query(row['query'], row['relevant_docs'], k)
                eval_result['query'] = row['query']
                query_results.append(eval_result)
            
            # Calcular promedios
            avg_metrics = {
                'avg_precision': np.mean([r['precision'] for r in query_results]),
                'avg_recall': np.mean([r['recall'] for r in query_results]),
                'avg_f1': np.mean([r['f1_score'] for r in query_results]),
                'mean_ap': np.mean([r['average_precision'] for r in query_results]),
                'avg_ndcg': np.mean([r['ndcg'] for r in query_results])
            }
            
            results[k] = {
                'query_results': query_results,
                'average_metrics': avg_metrics
            }
            
            # Mostrar resultados parciales
            logger.info(f"\nResultados parciales para k={k}:")
            for metric, value in avg_metrics.items():
                logger.info(f"{metric}: {value:.4f}")
        
        self.evaluation_results = results
        return results
    
    def plot_results(self, output_dir: str = 'evaluation_results'):
        """Genera visualizaciones de los resultados."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Preparar datos para gráficos
        k_values = sorted(self.evaluation_results.keys())
        metrics = {
            'avg_precision': 'Precisión',
            'avg_recall': 'Recall',
            'avg_f1': 'F1-Score',
            'mean_ap': 'MAP',
            'avg_ndcg': 'nDCG'
        }
        
        # Gráfico de líneas para métricas promedio
        plt.figure(figsize=(12, 6))
        for metric_key, metric_name in metrics.items():
            values = [self.evaluation_results[k]['average_metrics'][metric_key] for k in k_values]
            plt.plot(k_values, values, marker='o', label=metric_name)
        
        plt.title('Métricas promedio vs. K')
        plt.xlabel('K (número de resultados)')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_dir}/average_metrics.png')
        plt.close()
        
        # Guardar valores en CSV
        metrics_df = pd.DataFrame({
            'k': k_values,
            **{metric_name: [self.evaluation_results[k]['average_metrics'][metric_key] 
                           for k in k_values]
               for metric_key, metric_name in metrics.items()}
        })
        metrics_df.to_csv(f'{output_dir}/metrics_by_k.csv', index=False)
    
    def save_results(self, output_dir: str = 'evaluation_results'):
        """Guarda los resultados de la evaluación."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar resultados detallados en formato JSON
        detailed_results = {
            str(k): {
                'average_metrics': results['average_metrics'],
                'query_results': [
                    {**qr, 'query': str(qr['query'])}
                    for qr in results['query_results']
                ]
            }
            for k, results in self.evaluation_results.items()
        }
        
        with open(f'{output_dir}/detailed_results_{timestamp}.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # Crear DataFrame con resumen
        summary_rows = []
        for k, results in self.evaluation_results.items():
            row = {'k': k, **results['average_metrics']}
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(f'{output_dir}/evaluation_summary_{timestamp}.csv', index=False)
        
        logger.info(f"Resultados guardados en {output_dir}")
    
    def print_summary(self):
        """Imprime un resumen de la evaluación."""
        print("\nResumen de la evaluación del sistema")
        print("=" * 80)
        
        for k, results in sorted(self.evaluation_results.items()):
            metrics = results['average_metrics']
            print(f"\nResultados para k={k}:")
            print("-" * 40)
            print(f"Precisión promedio: {metrics['avg_precision']:.4f}")
            print(f"Recall promedio: {metrics['avg_recall']:.4f}")
            print(f"F1-Score promedio: {metrics['avg_f1']:.4f}")
            print(f"MAP: {metrics['mean_ap']:.4f}")
            print(f"nDCG promedio: {metrics['avg_ndcg']:.4f}")

def main():
    try:
        # Cargar motor de búsqueda
        from search_index import SearchEngine
        search_engine = SearchEngine(
            corpus_path='../data/processed/reuters_preprocessed_clean.csv',
            index_path='../data/processed/inverted_index.csv'
        )
        
        # Crear evaluador
        evaluator = SystemEvaluator(
            search_engine=search_engine,
            test_queries_path='../data/processed/test_queries.csv'
        )
        
        # Realizar evaluación
        evaluator.evaluate_system(k_values=[5, 10, 20])
        
        # Generar visualizaciones y guardar resultados
        evaluator.plot_results()
        evaluator.save_results()
        
        # Mostrar resumen
        evaluator.print_summary()
        
    except Exception as e:
        logger.error(f"Error durante la evaluación: {str(e)}")
        raise

if __name__ == "__main__":
    main()
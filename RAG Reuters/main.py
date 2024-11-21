from src.preprocessing import preprocess_corpus

if __name__ == "__main__":
    # Directorios y archivos
    input_dir = "data/reuters/training"  # Directorio con los archivos de entrenamiento
    output_dir = "data/processed/training"  # Directorio donde se guardarán los archivos procesados
    stopword_file = "data/reuters/stopwords.txt"  # Archivo de stopwords personalizadas

    # Ejecutar preprocesamiento
    preprocess_corpus(input_dir, output_dir, stopword_file)

    print("Preprocesamiento completado.")

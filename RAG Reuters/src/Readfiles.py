import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
import sklearn as sk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Tuple

def read_reuters_files(directory_path='../data/reuters/test'):
    documents = []
    try:
        filenames = sorted(os.listdir(directory_path))
        for filename in filenames:
            file_path = os.path.join(directory_path, filename)
            
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                        content = file.read()
                        documents.append((filename, content))
                except Exception as e:
                    print(f"Error reading file {filename}: {str(e)}")
    except Exception as e:
        print(f"Error accessing directory or reading files: {str(e)}")
        
    return documents

def load_stop_words(file_path='../data/reuters/stopwords.txt'):
    stop_words = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Agregar cada palabra al conjunto
                stop_words.add(line.strip().lower())
    except Exception as e:
        print(f"Error loading stop words: {e}")
        print(stop_words)
    return stop_words

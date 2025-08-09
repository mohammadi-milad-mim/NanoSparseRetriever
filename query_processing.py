# query_processing.py
import json
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class Query:
    def __init__(self, query_id, text):
        self.query_id = query_id
        self.text = text
        self.tokens = []
        self.term_weights = {}

    def preprocess(self, stop_words=None):
        """
        Preprocess the query text:
        - Tokenization
        - Stop-word removal
        - Stemming using Porter Stemmer
        """
        self.tokens = preprocess_text(self.text, stop_words)
        self.term_weights = {term: 1.0 for term in self.tokens}  # Assign weight 1.0 to original terms

    def expand(self, expansion_terms):
        """
        Expand the query with additional terms.
        Parameters:
        - expansion_terms: Dictionary of terms and their weights.
        """
        for term, weight in expansion_terms.items():
            if term in self.term_weights:
                self.term_weights[term] += weight
            else:
                self.term_weights[term] = weight
                self.tokens.append(term)  # Add new term to tokens

def preprocess_text(text, stop_words=None):
    """
    Preprocess the input text and return a list of tokens.
    """
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Lowercasing
    tokens = [token.lower() for token in tokens]
    
    # Stop-word removal
    if stop_words is None:
        stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens

def load_queries(file_path, dataset_type='type1'):
    """
    Parse the query file and return a list of Query objects.
    """
    if dataset_type == 'type1':
        return load_queries_type1(file_path)
    elif dataset_type == 'type2':
        return load_queries_type2(file_path)
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

def load_queries_type1(file_path):
    """
    Load queries from dataset type 1.
    """
    queries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for query_data in data.get('queries', []):
            query_id = query_data.get('number')
            text = query_data.get('text')
            query = Query(query_id, text)
            queries.append(query)
    return queries

def load_queries_type2(file_path):
    """
    Load queries from dataset type 2.
    """
    queries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Split into individual topics
        topic_texts = re.findall(r'<DOC\s*(\d+)>(.*?)</DOC>', content, re.DOTALL)
        for topic_id, topic_content in topic_texts:
            # The content may have terms each on a new line
            terms = topic_content.strip().split()
            text = ' '.join(terms)
            query = Query(topic_id.strip(), text)
            queries.append(query)
    return queries
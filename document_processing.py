# document_processing.py
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class Document:
    def __init__(self, doc_id, file_id, text, metadata=None):
        self.doc_id = doc_id
        self.file_id = file_id
        self.text = text
        self.tokens = []
        self.metadata = metadata or {}  # Store additional metadata

    def preprocess(self, stop_words=None):
        """
        Preprocess the document text:
        - Tokenization
        - Lowercasing
        - Stop-word removal
        - Stemming
        """
        self.tokens = preprocess_text(self.text, stop_words)

def preprocess_text(text, stop_words=None):
    """
    Preprocess the input text and return a list of tokens.
    Steps:
    - Remove non-alphabetic characters
    - Tokenization
    - Lowercasing
    - Stop-word removal
    - Stemming using Porter Stemmer
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

def load_stopwords(file_path=None):
    """
    Load stopwords from a file or use NLTK's default stopword list.
    """
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as f:
            stop_words = set(line.strip() for line in f if line.strip())
    else:
        stop_words = set(stopwords.words('english'))
    return stop_words

def load_documents(file_path, dataset_type='type1'):
    """
    Parse the corpus file and return a list of Document objects.
    """
    if dataset_type == 'type1':
        return load_documents_type1(file_path)
    elif dataset_type == 'type2':
        return load_documents_type2(file_path)
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

def load_documents_type1(file_path):
    """
    Load documents from dataset type 1.
    """
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        doc_text = ''
        inside_doc = False
        for line in f:
            line = line.strip()
            if line == '<DOC>':
                inside_doc = True
                doc_text = ''
            elif line == '</DOC>':
                inside_doc = False
                doc = parse_document_type1(doc_text)
                documents.append(doc)
            elif inside_doc:
                doc_text += line + '\n'
    return documents

def parse_document_type1(doc_text):
    """
    Parse a single document text block from dataset type 1 and return a Document object.
    """
    doc_id = extract_tag_content(doc_text, 'DOCNO')
    file_id = extract_tag_content(doc_text, 'FILEID')
    first = extract_tag_content(doc_text, 'FIRST')
    second = extract_tag_content(doc_text, 'SECOND')
    head = extract_tag_content(doc_text, 'HEAD')
    dateline = extract_tag_content(doc_text, 'DATELINE')
    text = extract_tag_content(doc_text, 'TEXT')
    metadata = {
        'first': first,
        'second': second,
        'head': head,
        'dateline': dateline
    }
    return Document(doc_id, file_id, text, metadata)

def load_documents_type2(file_path):
    """
    Load documents from dataset type 2.
    """
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Split the file into individual documents
        doc_texts = re.findall(r'<DOC>(.*?)</DOC>', content, re.DOTALL)
        for doc_text in doc_texts:
            doc = parse_document_type2(doc_text)
            if doc:
                documents.append(doc)
    return documents

def parse_document_type2(doc_text):
    """
    Parse a single document text block from dataset type 2 and return a Document object.
    """
    doc_id = extract_tag_content(doc_text, 'DOCNO')
    if not doc_id:
        return None  # Skip if DOCNO is missing
    doc_id = doc_id.strip()  # Remove any extra whitespace
    file_id = extract_tag_content(doc_text, 'DOCID').strip()
    text = extract_tag_content(doc_text, 'TEXT')
    # Extract text within <P> tags
    text = extract_text_from_p_tags(text)
    # We can also extract other metadata if needed
    metadata = {
        'source': extract_text_from_p_tags(extract_tag_content(doc_text, 'SOURCE')),
        'date': extract_text_from_p_tags(extract_tag_content(doc_text, 'DATE')),
        'section': extract_text_from_p_tags(extract_tag_content(doc_text, 'SECTION')),
        'length': extract_text_from_p_tags(extract_tag_content(doc_text, 'LENGTH')),
        'headline': extract_text_from_p_tags(extract_tag_content(doc_text, 'HEADLINE')),
        'byline': extract_text_from_p_tags(extract_tag_content(doc_text, 'BYLINE')),
        'type': extract_text_from_p_tags(extract_tag_content(doc_text, 'TYPE')),
    }
    return Document(doc_id, file_id, text, metadata)

def extract_text_from_p_tags(text):
    """
    Extract text from within <P> tags.
    """
    paragraphs = re.findall(r'<P>\s*(.*?)\s*</P>', text, re.DOTALL)
    content = ' '.join(paragraphs)
    return content

def extract_tag_content(text, tag):
    """
    Extract content between specific XML-like tags.
    """
    pattern = f'<{tag}>(.*?)</{tag}>'
    matches = re.findall(pattern, text, re.DOTALL)
    # Concatenate multiple occurrences of the same tag
    content = ' '.join(match.strip() for match in matches)
    return content
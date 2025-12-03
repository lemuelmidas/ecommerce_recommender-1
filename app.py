from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from pinecone import Pinecone
import pytesseract
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from sentence_transformers import SentenceTransformer
import io
import base64

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize models and databases (placeholders)
class Config:
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', 'your-api-key')
    PINECONE_INDEX_NAME = 'products-index'
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Global variables for models
embedding_model = None
pinecone_index = None
cnn_model = None
product_data = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_models():
    """Initialize all models and connections"""
    global embedding_model, pinecone_index, cnn_model, product_data
    
    # Load embedding model
    embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
    
    # Initialize Pinecone
    try:
        pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        pinecone_index = pc.Index(Config.PINECONE_INDEX_NAME)
    except Exception as e:
        print(f"Pinecone initialization error: {e}")
    
    # Load CNN model
    try:
        cnn_model = keras.models.load_model('models/cnn_product_classifier.h5')
    except:
        print("CNN model not found. Train the model first.")
    
    # Load product data
    try:
        product_data = pd.read_csv('data/cleaned_products.csv')
    except:
        print("Product data not found.")

def clean_dataset(df):
    """Task 1: Clean the e-commerce dataset"""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna({
        'product_name': 'Unknown',
        'description': '',
        'price': 0.0,
        'category': 'Uncategorized'
    })
    
    # Standardize formats
    df['product_name'] = df['product_name'].str.strip().str.title()
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
    
    return df

def create_embeddings(text):
    """Generate embeddings for text"""
    return embedding_model.encode(text).tolist()

def query_vector_db(query_vector, top_k=5):
    """Query Pinecone vector database"""
    try:
        results = pinecone_index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        return results
    except Exception as e:
        print(f"Vector DB query error: {e}")
        return None

def sanitize_query(query):
    """Safeguard against bad queries and sensitive data"""
    # Remove potentially harmful patterns
    harmful_patterns = ['delete', 'drop', 'truncate', 'exec', 'script']
    query_lower = query.lower()
    
    for pattern in harmful_patterns:
        if pattern in query_lower:
            return None, "Invalid query detected"
    
    # Length check
    if len(query) > 500:
        return None, "Query too long"
    
    return query, None

def generate_natural_response(products, query):
    """Generate natural language response"""
    if not products:
        return "I couldn't find any products matching your query. Please try different keywords."
    
    response = f"Based on your query, I found {len(products)} products that might interest you. "
    response += f"The top recommendation is {products[0]['name']} at ${products[0]['price']:.2f}."
    
    return response

def extract_text_from_image(image_path):
    """Task 4: OCR functionality"""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def predict_product_from_image(image_path):
    """Task 6: CNN prediction"""
    try:
        # Load and preprocess image
        img = keras.preprocessing.image.load_img(
            image_path, 
            target_size=(224, 224)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Predict
        predictions = cnn_model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Load class names
        with open('models/class_names.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        
        return {
            'class': class_names[predicted_class],
            'confidence': confidence
        }
    except Exception as e:
        return {'error': str(e)}

# ==================== ENDPOINTS ====================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-query')
def text_query_page():
    return render_template('text_query.html')

@app.route('/image-query')
def image_query_page():
    return render_template('image_query.html')

@app.route('/product-image')
def product_image_page():
    return render_template('product_image.html')

# ENDPOINT 1: Product Recommendation Service
@app.route('/api/recommend', methods=['POST'])
def recommend_products():
    """Handle natural language queries for product recommendations"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        # Sanitize query
        clean_query, error = sanitize_query(query)
        if error:
            return jsonify({'error': error}), 400
        
        # Generate embedding
        query_vector = create_embeddings(clean_query)
        
        # Query vector database
        results = query_vector_db(query_vector, top_k=5)
        
        if not results or not results.matches:
            return jsonify({
                'products': [],
                'response': 'No products found matching your query.'
            })
        
        # Format products
        products = []
        for match in results.matches:
            products.append({
                'id': match.id,
                'name': match.metadata.get('name', 'Unknown'),
                'description': match.metadata.get('description', ''),
                'price': match.metadata.get('price', 0.0),
                'category': match.metadata.get('category', ''),
                'similarity_score': float(match.score)
            })
        
        # Generate natural language response
        response_text = generate_natural_response(products, query)
        
        return jsonify({
            'products': products,
            'response': response_text,
            'query': clean_query
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ENDPOINT 2: OCR-Based Query Processing
@app.route('/api/ocr-query', methods=['POST'])
def ocr_query():
    """Process handwritten queries from images"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text using OCR
        extracted_text = extract_text_from_image(filepath)
        
        if not extracted_text:
            return jsonify({'error': 'No text found in image'}), 400
        
        # Process as normal query
        clean_query, error = sanitize_query(extracted_text)
        if error:
            return jsonify({'error': error}), 400
        
        # Generate embedding and query
        query_vector = create_embeddings(clean_query)
        results = query_vector_db(query_vector, top_k=5)
        
        products = []
        if results and results.matches:
            for match in results.matches:
                products.append({
                    'id': match.id,
                    'name': match.metadata.get('name', 'Unknown'),
                    'description': match.metadata.get('description', ''),
                    'price': match.metadata.get('price', 0.0),
                    'category': match.metadata.get('category', ''),
                    'similarity_score': float(match.score)
                })
        
        response_text = generate_natural_response(products, extracted_text)
        
        # Cleanup
        os.remove(filepath)
        
        return jsonify({
            'extracted_text': extracted_text,
            'products': products,
            'response': response_text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ENDPOINT 3: Image-Based Product Detection
@app.route('/api/detect-product', methods=['POST'])
def detect_product():
    """Identify products from images using CNN"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # CNN prediction
        prediction = predict_product_from_image(filepath)
        
        if 'error' in prediction:
            return jsonify({'error': prediction['error']}), 500
        
        # Query vector database for similar products
        product_class = prediction['class']
        query_vector = create_embeddings(product_class)
        results = query_vector_db(query_vector, top_k=5)
        
        products = []
        if results and results.matches:
            for match in results.matches:
                products.append({
                    'id': match.id,
                    'name': match.metadata.get('name', 'Unknown'),
                    'description': match.metadata.get('description', ''),
                    'price': match.metadata.get('price', 0.0),
                    'category': match.metadata.get('category', ''),
                    'similarity_score': float(match.score)
                })
        
        response_text = f"Detected product: {product_class} (confidence: {prediction['confidence']:.2%}). "
        response_text += generate_natural_response(products, product_class)
        
        # Cleanup
        os.remove(filepath)
        
        return jsonify({
            'detected_class': product_class,
            'confidence': prediction['confidence'],
            'products': products,
            'response': response_text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Utility endpoints
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'embedding_model_loaded': embedding_model is not None,
        'cnn_model_loaded': cnn_model is not None,
        'pinecone_connected': pinecone_index is not None
    })

if __name__ == '__main__':
    print("Initializing models...")
    initialize_models()
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
import json
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
from flask import Flask, request, jsonify
from google.cloud import storage
import os
import logging

app = Flask(__name__)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load model and processor once during cold start
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Placeholder for image embeddings
image_embeddings = None

# Environment variables for GCS
BUCKET_NAME = os.getenv('BUCKET_NAME', 'lolify_embeddings')
BLOB_NAME = os.getenv('BLOB_NAME', 'image_embeddings.json')

def calculate_cosine_similarity(vec1, vec2):
    """Calculate the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def load_image_embeddings():
    """Load image embeddings from the JSON file in GCS."""
    global image_embeddings
    if image_embeddings is None:
        # Initialize the GCS client
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(BLOB_NAME)
        
        try:
            logging.info(f"Downloading {BLOB_NAME} from bucket {BUCKET_NAME}.")
            image_embeddings_json = blob.download_as_text()
            image_embeddings = json.loads(image_embeddings_json)
            logging.info("Image embeddings loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading image embeddings: {e}")
            raise e
    return image_embeddings

def create_prompt_embedding(prompt):
    """Create an embedding for the given prompt using CLIP."""
    inputs = processor(text=[prompt], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features.cpu().numpy().flatten()

def is_similar(embedding1, embedding2, threshold=0.9):
    """Check if the cosine similarity between two embeddings exceeds a given threshold."""
    return calculate_cosine_similarity(embedding1, embedding2) > threshold

def calculate_inter_template_similarities(ranked_templates, image_embeddings):
    """Calculate cosine similarities among the templates themselves."""
    inter_template_similarities = {}
    
    for i, template_1 in enumerate(ranked_templates):
        for template_2 in ranked_templates[i+1:]:
            embedding_1 = image_embeddings.get(template_1)
            embedding_2 = image_embeddings.get(template_2)
            if embedding_1 is not None and embedding_2 is not None:
                similarity = calculate_cosine_similarity(embedding_1, embedding_2)
                inter_template_similarities[(template_1, template_2)] = similarity
    
    return inter_template_similarities

@app.route('/ranktemplates', methods=['POST'])
def rank_templates():
    data = request.get_json()
    prompt = data.get('prompt')
    num_templates = data.get('num_templates', 1)

    if not prompt:
        return jsonify({"error": "Invalid input, 'prompt' is required"}), 400

    try:
        prompt_embedding = create_prompt_embedding(prompt)
        image_embeddings = load_image_embeddings()
    except Exception as e:
        logging.error(f"Error processing embeddings: {e}")
        return jsonify({"error": "Failed to load embeddings"}), 500

    similarities = []

    # Calculate similarity of the prompt embedding to all image embeddings
    for image_name, embedding in image_embeddings.items():
        similarity = calculate_cosine_similarity(prompt_embedding, embedding)
        similarities.append((image_name, similarity, embedding))

    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Collect templates, ensuring no high similarity between selected templates
    ranked_templates = []
    selected_embeddings = []

    for image_name, _, embedding in similarities:
        if len(ranked_templates) >= num_templates:
            break

        # Check if the current template is too similar to any of the already selected ones
        if all(not is_similar(embedding, selected_embedding) for selected_embedding in selected_embeddings):
            ranked_templates.append(image_name)
            selected_embeddings.append(embedding)

    # Calculate inter-template similarities
    inter_template_similarities = calculate_inter_template_similarities(ranked_templates, image_embeddings)

    # Log the calculated similarities
    for template_pair, similarity in inter_template_similarities.items():
        logging.info(f"Similarity between {template_pair[0]} and {template_pair[1]}: {similarity}")

    return jsonify({"ranked_templates": ranked_templates})

# Entry point for Google Cloud Functions
def main(request):
    return app(request)

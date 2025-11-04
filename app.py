# app.py
import flask
from flask import Flask, request, jsonify
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re
import sys
import os

# ---------- Config ----------
# Note: These models are downloaded from Hugging Face and require an internet connection on first run.
# For production, you should pre-download the models or use smaller/local models.
IPC_CSV = "ipc_paragraphs.csv"
EMBED_MODEL = "all-MiniLM-L6-v2"
EXPLAIN_MODEL = "google/flan-t5-base"
MAX_TOKENS = 60
# ----------------------------

app = Flask(__name__)

# Global variables for models and data
df = None
embedder = None
corpus_embeddings = None
qa_pipeline = None
corpus = None

def contains_violent_intent(text):
    """Simple check to block violent instructions."""
    violent_patterns = [
        r"\bkill\b", r"\bmurder\b", r"\bassassinat", r"\bblow up\b", r"\bpoison\b",
        r"\bhow to (kill|murder|hurt|harm|attack)\b"
    ]
    t = text.lower()
    return any(re.search(p, t) for p in violent_patterns)

def load_models_and_data():
    """Loads all models and data once when the app starts."""
    global df, embedder, corpus_embeddings, qa_pipeline, corpus
    
    # 1. Load Data
    try:
        if not os.path.exists(IPC_CSV):
            print(f"ERROR: {IPC_CSV} not found. Please ensure it is in the same directory as app.py.")
            sys.exit(1)

        df = pd.read_csv(IPC_CSV)
        if "section" not in df.columns or "text" not in df.columns:
            raise ValueError("CSV must contain 'section' and 'text' columns.")
        corpus = df["text"].tolist()
        print("✅ Data loaded.")
    except Exception as e:
        print(f"Error loading '{IPC_CSV}': {e}")
        sys.exit(1)

    # 2. Load Embedder for semantic search and calculate embeddings
    try:
        print("Loading Sentence Transformer...")
        embedder = SentenceTransformer(EMBED_MODEL)
        corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
        print("✅ Embeddings calculated.")
    except Exception as e:
        print(f"Error loading Sentence Transformer: {e}")
        # In a real app, you might want to handle this gracefully
        sys.exit(1)

    # 3. Load Text generation / explanation pipeline (deterministic)
    try:
        print("Loading Flan-T5 model...")
        # Note: device=-1 for CPU, 0 for first GPU
        qa_pipeline = pipeline(
            "text2text-generation",
            model=EXPLAIN_MODEL,
            device=-1
        )
        print("✅ Flan-T5 pipeline loaded.")
    except Exception as e:
        print(f"Error loading Flan-T5: {e}")
        sys.exit(1)

# Before first request, load models and data
with app.app_context():
    load_models_and_data()

# ----------------- FLASK API ENDPOINT -----------------

@app.route('/ask', methods=['POST'])
def ask_legalbot():
    """
    API endpoint to query the legal chatbot.
    Expects a JSON payload: {"query": "what is the punishment for theft?"}
    """
    
    # 1. Get input query
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
    except:
        return jsonify({"error": "Invalid JSON format or missing 'query' field."}), 400

    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    # 2. Safety Check (from ipc_chatbot.py)
    if contains_violent_intent(query):
        return jsonify({
            "section": "SAFETY BLOCKED",
            "explanation": "I can't help with instructions to harm people. If you or someone is in danger, contact local authorities or a trusted person immediately."
        }), 403

    # 3. Semantic Search
    try:
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)
        best = hits[0][0]
        section_idx = best["corpus_id"]
        section_code = df.iloc[section_idx]["section"]
        section_text = df.iloc[section_idx]["text"]
    except Exception as e:
        return jsonify({"error": f"Search failed: {e}"}), 500

    # 4. Build Prompt for Explainer Model
    prompt = (
        "You are LegalBot, an Indian law assistant. "
        "Explain the following Indian Penal Code section in plain, simple English for common people. "
        "Keep it short (about 30–50 words). No extra commentary, no legal citations, just the plain explanation and the punishment if any.\n\n"
        f"IPC SECTION: {section_code}\n{section_text}\n\nAnswer:"
    )

    # 5. Generate Explanation
    try:
        gen = qa_pipeline(
            prompt,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            num_return_sequences=1,
            repetition_penalty=1.6,
            truncation=True
        )
        response_text = gen[0].get("generated_text", "").strip()

        # Final trimming/safety (from ipc_chatbot.py)
        sentences = re.split(r'(?<=[.!?])\s+', response_text)
        short = " ".join(sentences[:3])
        words = short.split()
        if len(words) > 121:
             short = " ".join(words[:121]).rstrip() + "..."
             
        explanation = short
    except Exception as e:
        # Fallback to the raw section text if generation fails
        explanation = f"Generation failed. Related IPC Text: {section_text}"
        app.logger.error(f"Generation error: {e}")


    # 6. Return JSON Response
    return jsonify({
        "query": query,
        "related_section": section_code,
        "explanation": explanation
    })

if __name__ == '__main__':
    # Flask will automatically use the PORT environment variable if running in a cloud environment
    port = int(os.environ.get('PORT', 5000))
    # Note: Use 0.0.0.0 for external access in deployment
    app.run(host='127.0.0.1', port=port, debug=True, threaded=False)
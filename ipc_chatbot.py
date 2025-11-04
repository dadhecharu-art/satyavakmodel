# ipc_chatbot.py
# Clean, deterministic legal Q&A over IPC sections using semantic search + Flan-T5
# Safety: refuses violent/harm instructions.

from transformers import pipeline
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re
import sys

# ---------- Config ----------
IPC_CSV = "ipc_paragraphs.csv"
EMBED_MODEL = "all-MiniLM-L6-v2"       # fast semantic embedder
EXPLAIN_MODEL = "google/flan-t5-base"  # explain/generation model (deterministic settings)
MAX_TOKENS = 60                        # produce short replies (~40-60 tokens)
# ----------------------------

def contains_violent_intent(text):
    # simple check to block violent instructions (refuse to help)
    violent_patterns = [
        r"\bkill\b", r"\bmurder\b", r"\bassassinat", r"\bblow up\b", r"\bpoison\b",
        r"\bhow to (kill|murder|hurt|harm|attack)\b"
    ]
    t = text.lower()
    return any(re.search(p, t) for p in violent_patterns)

def load_data():
    try:
        df = pd.read_csv(IPC_CSV)
        if "section" not in df.columns or "text" not in df.columns:
            raise ValueError("CSV must contain 'section' and 'text' columns.")
        return df
    except Exception as e:
        print(f"Error loading '{IPC_CSV}': {e}")
        sys.exit(1)

def main():
    print("Loading data and models (this may take a moment)...")
    df = load_data()

    # Embedder for semantic search
    embedder = SentenceTransformer(EMBED_MODEL)
    corpus = df["text"].tolist()
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    # Text generation / explanation pipeline (deterministic)
    qa_pipeline = pipeline(
        "text2text-generation",
        model=EXPLAIN_MODEL,
        device=-1  # CPU; change to 0 for GPU if available
    )

    print("✅ LegalBot is ready! Ask me anything about Indian Penal Code.\n(Type 'exit' to quit.)\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nLegalBot: Goodbye! 👋")
            break

        if not query:
            continue
        if query.lower() in ["exit", "quit", "bye"]:
            print("LegalBot: Goodbye! 👋")
            break

        # Safety: refuse violent/illegal instructions to commit harm
        if contains_violent_intent(query):
            print("LegalBot: I can't help with instructions to harm people. If you or someone is in danger, contact local authorities or a trusted person immediately.")
            continue

        # Semantic search to find best-matching IPC section
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)
        best = hits[0][0]  # best match
        section_idx = best["corpus_id"]
        section_code = df.iloc[section_idx]["section"]
        section_text = df.iloc[section_idx]["text"]

        # Build a concise prompt for the explainer model
        prompt = (
            "You are LegalBot, an Indian law assistant. "
            "Explain the following Indian Penal Code section in plain, simple English for common people. "
            "Keep it short (about 30–50 words). No extra commentary, no legal citations, just the plain explanation and the punishment if any.\n\n"
            f"IPC SECTION: {section_code}\n{section_text}\n\nAnswer:"
        )

        # Generate — deterministic settings to avoid long nonsense
        gen = qa_pipeline(
            prompt,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,            # deterministic
            num_return_sequences=1,
            repetition_penalty=1.6,     # avoid repeats
            truncation=True
        )

        # Extract text safely
        response_text = gen[0].get("generated_text", "").strip()

        # Final safety / length enforcement: cut to first 3 sentences or ~50 words
        sentences = re.split(r'(?<=[.!?])\s+', response_text)
        short = " ".join(sentences[:3])
        # Truncate to ~50 words
        words = short.split()
        if len(words) > 121:
            short = " ".join(words[:121]).rstrip() + "..."

        # Print once (no duplicate prints)
        print(f"\n📘 Related IPC Section: {section_code}")
        print(f"LegalBot: {short}\n")

if __name__ == "__main__":
    main()

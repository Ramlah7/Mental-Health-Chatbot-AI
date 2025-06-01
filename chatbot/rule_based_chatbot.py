# File: chatbot/rule_based_chatbot.py
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch,pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
def here(*p): return str(ROOT.joinpath(*p).resolve().as_posix())

EMBED_MODEL    = "all-mpnet-base-v2"
INDEX_PATH     = here("scripts", "models", "faiss_index", "index.bin")
RESPONSES_PATH = here("scripts", "models", "faiss_index", "responses.csv")
FT_MODEL_DIR   = here("scripts", "models", "mindmate_dialo")
SIM_THRESHOLD  = 0.78

# Load SBERT embedder and FAISS index
_embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
_index    = faiss.read_index(INDEX_PATH)
_responses = pd.read_csv(RESPONSES_PATH)["bot_reply"].tolist()


tok   = AutoTokenizer.from_pretrained(FT_MODEL_DIR)
tok.pad_token = tok.eos_token
_model = AutoModelForCausalLM.from_pretrained(FT_MODEL_DIR)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model.to(_DEVICE)

def generate_bot_reply(user_input: str, max_new_tokens: int = 80) -> str:
    """Return the bot reply by trying retrieval first, then generative fallback."""
    # Retrieval
    vec = _embedder.encode([user_input]).astype("float32")
    D, I = _index.search(vec, 1)
    score = float(D[0][0]); idx = int(I[0][0])
    if score >= SIM_THRESHOLD:
        return _responses[idx]

    # Generative fallback
    inputs = tok.encode(user_input + tok.eos_token, return_tensors="pt").to(_DEVICE)
    out    = _model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id,
        no_repeat_ngram_size=3,
        top_p=0.95,
        temperature=0.7,
    )
    reply = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    return reply.strip()

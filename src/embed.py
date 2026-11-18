import numpy as np
import pandas as pd
import ollama

MODEL = "nomic-embed-text"
PREFIX = "classification: "  # nomic-embed-text task prefix
BATCH = 64


def embed_texts(texts):
    out = []
    for i in range(0, len(texts), BATCH):
        batch = [PREFIX + t for t in texts[i:i + BATCH]]
        resp = ollama.embed(model=MODEL, input=batch)
        out.extend(resp["embeddings"])
        print(f"  {min(i + BATCH, len(texts))}/{len(texts)}", end="\r", flush=True)
    print()
    return np.array(out, dtype=np.float32)


def main():
    for name in ["train", "val", "test"]:
        df = pd.read_csv(f"data/processed/{name}.csv")
        print(f"{name}: {len(df)} messages")
        emb = embed_texts(df["text"].tolist())
        np.save(f"data/processed/emb_{name}.npy", emb)
        print(f"saved emb_{name}.npy shape={emb.shape}")


if __name__ == "__main__":
    main()

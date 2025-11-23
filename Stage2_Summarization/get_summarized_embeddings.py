import os
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = "text-embedding-3-small"

SAVE_EMB_DIR = "Stage2_Summarization/embeddings"
os.makedirs(SAVE_EMB_DIR, exist_ok=True)

def embed_text(t):
    if not t.strip():
        return None
    try:
        r = client.embeddings.create(model=EMBED_MODEL, input=t)
        return np.array(r.data[0].embedding, dtype=float)
    except:
        return None


METHOD_MAP = {
    "frequency": "frequency_summarized_segments.json",
    "tfidf": "tfidf_summarized_segments.json",
    "lsa": "lsa_summarized_segments.json",
    "lda": "lda_summarized_segments.json",
    "llm": "llm_stage_2_summarized_segments.json"
}

SEGMENT_RESULT_DIR = "Stage1_Segmentation/segment_result"
SUMMARY_DIR = "Stage2_Summarization/summarized_results"
LLM_SEGMENT_RESULT_DIR = "Stage2_Summarization/llm_all_summarized_results"

for file in os.listdir(SEGMENT_RESULT_DIR):

    if not file.endswith(".json"):
        continue

    video_id = file.split("segment_result_preprocessed_")[1].split(".json")[0]

    print(f"\n=== Processing {video_id} ===")

    # Stage1 segmentation (원본 텍스트 embedding)
    seg_path = os.path.join(SEGMENT_RESULT_DIR, file)
    seg_data = json.load(open(seg_path, "r", encoding="utf-8"))

    segments = seg_data["segments"]

    for seg in segments:
        seg_num = seg["seg_num"]
        text = seg.get("text", "")

        emb = embed_text(text)
        if emb is None:
            continue

        save_path = os.path.join(SAVE_EMB_DIR, f"{video_id}__seg{seg_num}_origin.npy")
        np.save(save_path, emb)

    # Stage2 chapter titles
    base_dir = os.path.join(SUMMARY_DIR, video_id)

    for method, summary_file in METHOD_MAP.items():
        path = os.path.join(base_dir, summary_file)
        if not os.path.exists(path):
            continue

        stage2 = json.load(open(path, "r", encoding="utf-8"))

        for seg in stage2["segments"]:
            seg_num = seg["seg_num"]

            title = seg.get("title", seg.get("text", ""))
            emb = embed_text(title)
            if emb is None:
                continue

            save_path = os.path.join(SAVE_EMB_DIR, f"{video_id}__seg{seg_num}_{method}.npy")
            np.save(save_path, emb)

print("Embedding saved to:", SAVE_EMB_DIR)

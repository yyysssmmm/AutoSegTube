from collections import Counter
import numpy as np
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
import os
import json


# ============================================================
# 1) Frequency-based (기존 동일)
# ============================================================
def frequency_based_summarization(segments: List[Dict], top_k=5) -> List[Dict]:
    ans = []

    for segment in segments:
        text = segment["text"].strip()
        words = text.split()
        counter = Counter(words)

        top_keywords = [w for w, c in counter.most_common(top_k)]

        title = " · ".join([w.capitalize() for w in top_keywords[:3]])
        ans.append({
            "seg_num": segment["seg_num"],
            "start_timestamp": segment["start_timestamp"],
            "title": title,
        })

    return ans


# ============================================================
# 2) TF-IDF 기반 (기존 동일)
# ============================================================
def tfidf_based_summarization(segments: List[Dict], top_k=5) -> List[Dict]:
    ans = []

    for segment in segments:
        text = segment["text"].strip()

        vec = TfidfVectorizer(max_features=2000)
        tfidf = vec.fit_transform([text])
        scores = tfidf.toarray()[0]

        pairs = list(zip(vec.get_feature_names_out(), scores))
        pairs.sort(key=lambda x: x[1], reverse=True)

        top_keywords = [w for w, s in pairs[:top_k]]
        title = " · ".join([w.capitalize() for w in top_keywords[:3]])

        ans.append({
            "seg_num": segment["seg_num"],
            "start_timestamp": segment["start_timestamp"],
            "title": title,
        })

    return ans


# ============================================================
# 3) ★ 개선된 LSA 기반 Summarization (진짜 Topic Modeling 구조)
# ============================================================
def lsa_based_summarization(segments: List[Dict], n_components=2, top_k=5):
    """
    Proper LSA:
    - 모든 segment 텍스트를 문서 집합으로 보고
    - TF-IDF → SVD → Topic 분석
    """

    texts = [seg["text"] for seg in segments]

    # 1) TF-IDF 전체 문서에 적용
    vectorizer = TfidfVectorizer(max_features=3000)
    tfidf_matrix = vectorizer.fit_transform(texts)  # shape = (num_segments, vocab)

    # 2) SVD로 잠재 토픽 추출
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    doc_topic = svd.fit_transform(tfidf_matrix)  # segment-topic 분포

    vocab = vectorizer.get_feature_names_out()
    topic_term = svd.components_  # (topic, vocab)

    # 3) 각 segment의 대표 토픽 선택
    segment_titles = []
    for i, seg in enumerate(segments):

        topic_idx = np.argmax(doc_topic[i])  # 가장 강한 토픽 하나 선택

        # 해당 topic의 단어 가중치 상위 k개
        top_indices = topic_term[topic_idx].argsort()[::-1][:top_k]
        top_keywords = [vocab[j] for j in top_indices]

        title = " · ".join([w.capitalize() for w in top_keywords[:3]])

        segment_titles.append({
            "seg_num": seg["seg_num"],
            "start_timestamp": seg["start_timestamp"],
            "title": title
        })

    return segment_titles, {
        "vectorizer": vectorizer,
        "doc_topic": doc_topic,
        "topic_term": topic_term,
        "vocab": vocab,
        "svd_model": svd
    }



# ============================================================
# 4) ★ LDA 기반 Summarization (pyLDAvis 시각화 준비 완료)
# ============================================================
def lda_based_summarization(segments: List[Dict], n_topics=3, top_k=5):
    """
    Latent Dirichlet Allocation 기반 Topic Modeling
    """

    texts = [seg["text"] for seg in segments]

    # 1) Bag-of-Words 벡터화
    vectorizer = CountVectorizer(max_features=3000, stop_words="english")
    bow_matrix = vectorizer.fit_transform(texts)

    # 2) LDA 모델링
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch"
    )

    doc_topic = lda.fit_transform(bow_matrix)  # (segment, topic)
    topic_term = lda.components_              # (topic, vocab)
    vocab = vectorizer.get_feature_names_out()

    # 3) Segment 제목 생성
    segment_titles = []
    for i, seg in enumerate(segments):
        topic_idx = np.argmax(doc_topic[i])

        top_indices = topic_term[topic_idx].argsort()[::-1][:top_k]
        top_keywords = [vocab[j] for j in top_indices]

        title = " · ".join([w.capitalize() for w in top_keywords[:3]])

        segment_titles.append({
            "seg_num": seg["seg_num"],
            "start_timestamp": seg["start_timestamp"],
            "title": title
        })

    # 4) pyLDAvis용 데이터도 함께 반환
    lda_metadata = {
        "vectorizer": vectorizer,
        "bow_matrix": bow_matrix,
        "lda_model": lda,
        "doc_topic": doc_topic,
        "topic_term": topic_term,
        "vocab": vocab
    }

    return segment_titles, lda_metadata



# ============================================================
# Main Execution
# ============================================================
# ============================================================
# Main Execution (수정됨 - pyLDAvis 복원 가능한 파일 저장)
# ============================================================
if __name__ == "__main__":
    SOURCE_DIR = "Stage1_Segmentation/segment_result"
    OUTPUT_DIR = "Stage2_Summarization/summarized_results"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for file in os.listdir(SOURCE_DIR):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(SOURCE_DIR, file), "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = data["segments"]

        # ---- 기존 방식 ----
        frequency_summ = data.copy()
        frequency_summ["segments"] = frequency_based_summarization(segments)
        
        tfidf_summ = data.copy()
        tfidf_summ["segments"] = tfidf_based_summarization(segments)

        # ---- 개선된 LSA ----
        lsa_summ = data.copy()
        lsa_summ["segments"], lsa_meta = lsa_based_summarization(segments)

        # ---- LDA ----
        lda_summ = data.copy()
        lda_summ["segments"], lda_meta = lda_based_summarization(segments)

        # save 결과
        base_name = file.replace(".json", "").split("segment_result_preprocessed_")[1]
        save_dir = os.path.join(OUTPUT_DIR, base_name)
        os.makedirs(save_dir, exist_ok=True)

        json.dump(frequency_summ, open(os.path.join(save_dir, "frequency_summarized_segments.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(tfidf_summ, open(os.path.join(save_dir, "tfidf_summarized_segments.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(lsa_summ, open(os.path.join(save_dir, "lsa_summarized_segments.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(lda_summ, open(os.path.join(save_dir, "lda_summarized_segments.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        # =====================================================
        # 🟩 pyLDAvis 복원용 파일 저장
        # =====================================================
        print(f"[INFO] Saving pyLDAvis metadata for {base_name}")

        # 1) Doc-Topic (num_docs x num_topics)
        np.save(os.path.join(save_dir, "lda_doc_topic.npy"), lda_meta["doc_topic"])

        # 2) Topic-Term (num_topics x vocab_size)
        np.save(os.path.join(save_dir, "lda_topic_term.npy"), lda_meta["topic_term"])

        # 3) Bag-of-Words Matrix (DTM)
        np.save(os.path.join(save_dir, "lda_bow_matrix.npy"), lda_meta["bow_matrix"].toarray())

        # 4) Vocabulary 저장
        vocab_list = lda_meta["vocab"].tolist()
        json.dump(vocab_list, open(os.path.join(save_dir, "lda_vocab.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        # (선택) vectorizer 설정도 저장해두면 완전 재현됨
        vectorizer_info = {
            "max_features": lda_meta["vectorizer"].max_features,
            "stop_words": lda_meta["vectorizer"].stop_words,
            "token_pattern": lda_meta["vectorizer"].token_pattern,
        }
        json.dump(vectorizer_info, open(os.path.join(save_dir, "lda_vectorizer_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        print(f"[OK] Saved: {save_dir}")

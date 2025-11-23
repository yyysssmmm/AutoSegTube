# 🎬 AutoSegTube: YouTube Transcript–Based Segmentation & Summarization Pipeline

본 프로젝트는 **YouTube 영상의 텍스트 트랜스크립트를 기반으로 자동 챕터링(Auto-Segmentation)** 및  
**다양한 방식의 자동 요약(Auto-Summarization)**을 수행하는 전체 파이프라인이다.

**Embedding → Preprocessing → Segmentation → Summarization → Visual Evaluation**  
까지의 모든 단계를 자동화하였다.

---

## 📌 Pipeline Overview

전체 프로세스는 아래 순서로 진행된다:

1. **video_id.txt 준비**
2. **Transcript 수집 (`fetch_transcript.py`)**
3. **Sentence Embedding 생성 (`generate_embeddings.py`)**
4. **Transcript 전처리 (`preprocess_transcript.py`)**
5. **세그먼트 추출 (Segmentation, `find_segment.py`)**
6. **요약 생성 (Summarization)**
7. **Summarized Embedding 생성 (`get_summarized_embeddings.py`)**
8. **정량·정성 시각화 분석 (`visualization.ipynb`)**

---

# 1️⃣ Transcript Fetching

`fetch_transcript.py` 실행 시 다음 구조로 저장됨:

```
transcript/
    └── {video_id}.json
```

JSON 파일 구조:

- `all_text`: 전체 영상 텍스트
- `sentences`: `[{"start_timestamp": ..., "text": ...}, ... ]`
- `gt`: 동일 구조(ground truth chapter timestamp)

---

# 2️⃣ Embedding Generation

`generate_embeddings.py` 실행 시:

```
embedded_transcripts/
    └── {video_id}.json
```

`sentences`·`gt` 리스트의 각 요소에:

```json
{
  "start_timestamp": ...,
  "text": "...",
  "embedding": [...]
}
```

형태로 embedding 추가됨.

---

# 3️⃣ Preprocessing (NER 기반 보호 + Stopword whitelist 반영)

최종 전처리 단계 순서는 다음과 같이 재설계됨:

1. Contract expansion  
2. 비언어 표현 및 구두점 제거  
3. **NER 기반 고유명사 보호**  
4. Stopword 제거 (단, whitelist 단어 유지)  
5. Lemmatization (고유명사 제외)

---

# 4️⃣ Stage 1 — Segmentation

`find_segment.py` 실행 후:

```
Stage1_Segmentation/segment_result/
    └── {video_id}_{k_thr}.json
```

JSON 구조:

```json
{
  "k": ...,
  "threshold": ...,
  "segments": [
    {
      "seg_num": ...,
      "start_timestamp": ...,
      "text": "...",
      "embedding": [...]
    }
  ]
}
```

---

# 5️⃣ Stage 2 — Extractive Summarization

`ours_summarize_segment.py` 실행 시:

```
Stage2_Summarization/summarized_results/{video_id}/
    ├── frequency_summarized_segments.json
    ├── tfidf_summarized_segments.json
    ├── lsa_summarized_segments.json
    └── lda_summarized_segments.json
```

---

# 6️⃣ Stage 2 — LLM Summarization

LLM 기반 summarization 결과:

```
llm_stage_2_summarized_segments.json  
llm_stage_all_summarized_segments.json
```

---

# 7️⃣ Summarized Embedding Generation

```
Stage2_Summarization/embeddings/
    └── {video_id}__seg{n}_{method}.npy
```

---

# 8️⃣ Visualization & Evaluation

- Segmentation t-SNE  
- Segmentation timeline 비교  
- Summarization wordcloud  
- Summarization embedding t-SNE  
- LDA pyLDAvis topic 시각화  

---

# 🔍 한계점 & 개선방안

## Segmentation
- Sentence-level semantic drift  
- Hyperparameter 민감  
- Metric 필요  

### 개선
- Chunk-based embedding  
- 자동 평가 metric  
- Multi-stage segmentation  

## Summarization
- Extractive 방식의 한계  
- 표현 다양성 부족  

### 개선
- Keyword 개수 확장  
- LLM fine-tuning  
- Global context 반영  

---

AutoSegTube는 YouTube 영상의 자동 챕터링 및 요약을 위한 End-to-End 파이프라인이며,  
향후 VectorDB, supervised fine-tuning 등 확장 가능성을 지닌 프로젝트이다.

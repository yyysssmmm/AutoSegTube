# Conda 환경에서: pip install openai python-dotenv

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

env_loaded = load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"  # 저렴하고 성능이 좋은 모델

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def get_embedding(segment_dict_list, is_gt=False):
    empty_count = 0
    success_count = 0
    fail_count = 0

    embedded_data = []

    timestamp_key = "start_timestamp"
    text_key = "text"
    
    for i, item in enumerate(segment_dict_list):

        transcript_text = item[text_key]
        start_time = item[timestamp_key]

        if not transcript_text:
            # 빈 텍스트인 경우 빈 embedding 저장
            empty_count += 1
            embedded_data.append(
                {
                    timestamp_key: start_time,
                    text_key: transcript_text,
                    "embedding": [],
                }
            )
            continue

        try:
            # OpenAI API로 embedding 생성
            response = client.embeddings.create(
                model=EMBEDDING_MODEL, input=transcript_text
            )
            embedding = response.data[0].embedding
            embedding_dim = len(embedding)

            embedded_data.append(
                {
                    timestamp_key: start_time,
                    text_key: transcript_text,
                    "embedding": embedding,
                }
            )
            success_count += 1

            # 첫 번째 embedding 차원 정보 출력
            if i == 0:
                print(f"[INFO] Embedding 차원: {embedding_dim}")

            # 진행 상황 출력 (50개마다)
            if (i + 1) % 50 == 0:
                progress_pct = (i + 1) / len(sentences_data) * 100
                print(
                    f"  진행: {i + 1}/{len(sentences_data)} ({progress_pct:.1f}%) - 성공: {success_count}, 실패: {fail_count}, 빈 텍스트: {empty_count}"
                )

        except Exception as e:
            fail_count += 1
            error_msg = str(e)
            # 에러 메시지가 너무 길면 자르기
            if len(error_msg) > 100:
                error_msg = error_msg[:100] + "..."
            print(f"  [경고] 항목 {i+1}에서 embedding 생성 실패: {error_msg}")
            # 실패한 경우 빈 embedding 저장
            embedded_data.append(
                {
                    timestamp_key: start_time,
                    text_key: transcript_text,
                    "embedding": [],
                }
            )
            continue

    return embedded_data, empty_count, success_count, fail_count


INPUT_DIR = "preprocessed_transcripts"
OUTPUT_DIR = "embedded_transcripts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

transcript_files = []
if os.path.exists(INPUT_DIR):
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".json"):
            transcript_files.append(filename)

# 각 트랜스크립트 파일 처리
for transcript_file in transcript_files:
    input_path = os.path.join(INPUT_DIR, transcript_file)
    output_filename = f"embedded_{transcript_file}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    ans = {}
    
    # JSON 파일 읽기
    with open(input_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    all_text = transcript_data.get("all_text", "")

    ans["all_text"] = all_text

    sentences_data = transcript_data.get("sentences", [])
    gt_data = transcript_data.get("gt", [])

    embedded_data, empty_count, success_count, fail_count = get_embedding(sentences_data)

    ans["sentences"] = embedded_data

    print(
        f"[INFO] sentence 임베딩 통계 - 성공: {success_count}, 실패: {fail_count}, 빈 텍스트: {empty_count}, 총: {len(sentences_data)}\n"
    )

    embedded_gt_data, empty_count, success_count, fail_count = get_embedding(gt_data, is_gt=True)

    ans["gt"] = embedded_gt_data

    print(
        f"[INFO] gt 임베딩 통계 - 성공: {success_count}, 실패: {fail_count}, 빈 텍스트: {empty_count}, 총: {len(sentences_data)}\n"
    )

    # 결과 저장
    print(f"[INFO] 결과 저장 중...")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ans, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] {output_filename} 파일 저장 완료.")

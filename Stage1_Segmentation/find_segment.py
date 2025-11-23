#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_segment.py - Hierarchical Video Segmentation Script

This script performs a two-step segmentation process:
1. Coarse-grained boundary detection (window level)
2. Fine-grained refinement (sentence level)
"""

import os
import json
import argparse
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_MODEL = "text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.4
K = 15
TOP_N_CANDIDATES = 5

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(filepath):
    """
    JSON 파일을 로드하고 sentences를 DataFrame으로 변환합니다.

    Args:
        filepath (str): JSON 파일 경로

    Returns:
        pd.DataFrame: sentences 데이터를 담은 DataFrame
    """
    try:
        logger.info(f"데이터 로드 중: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 새로운 형식: {"all_text": ..., "sentences": [...]}
        if isinstance(data, dict) and "sentences" in data:
            sentences = data["sentences"]
            logger.info("새로운 데이터 형식 감지: all_text와 sentences")
        elif isinstance(data, list):
            # 기존 형식 (하위 호환성)
            sentences = data
            logger.info("기존 데이터 형식 감지: 배열")
        else:
            raise ValueError(
                f"지원하지 않는 데이터 형식입니다. 예상 형식: {{'all_text': ..., 'sentences': [...]}} 또는 [...]"
            )

        if not sentences:
            raise ValueError("sentences 데이터가 비어있습니다.")

        # DataFrame으로 변환
        df = pd.DataFrame(sentences)
        logger.info(f"총 {len(df)}개의 문장을 로드했습니다.")

        return df

    except FileNotFoundError:
        logger.error(f"파일을 찾을 수 없습니다: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
        raise


def get_embedding(text, client):
    """
    텍스트를 임베딩 벡터로 변환합니다.

    Args:
        text (str): 임베딩할 텍스트
        client (OpenAI): OpenAI 클라이언트

    Returns:
        list: 임베딩 벡터
    """
    try:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"임베딩 생성 실패: {str(e)}")
        raise


def create_windows(df, k, client):
    """
    DataFrame을 k개씩 non-overlapping 윈도우로 나누고 각 윈도우의 임베딩을 생성합니다.

    Args:
        df (pd.DataFrame): 문장 데이터가 담긴 DataFrame
        k (int): 윈도우 크기 (step_size)
        client (OpenAI): OpenAI 클라이언트

    Returns:
        list: 처리된 윈도우 정보를 담은 딕셔너리 리스트
    """
    try:
        logger.info(f"윈도우 생성 중... (윈도우 크기: {k})")
        processed_windows = []

        # k개씩 non-overlapping 윈도우 생성
        for i in range(0, len(df), k):
            window_index = i // k
            window_df = df.iloc[i : i + k]

            if len(window_df) == 0:
                continue

            # k개의 transcript를 하나의 긴 텍스트로 합치기
            transcripts = window_df["text"].tolist()
            long_text = " ".join(transcripts)

            # 임베딩 생성
            logger.info(
                f"  윈도우 {window_index + 1} 처리 중... ({len(window_df)}개 문장)"
            )
            embedding = get_embedding(long_text, client)

            # 첫 번째와 마지막 문장 정보
            first_sentence = transcripts[0] if transcripts else ""
            last_sentence = transcripts[-1] if transcripts else ""
            first_timestamp = (
                window_df.iloc[0]["start_timestamp"] if len(window_df) > 0 else 0
            )
            start_index = i  # DataFrame에서의 시작 인덱스
            end_index = min(i + k, len(df))  # DataFrame에서의 끝 인덱스

            # 윈도우 정보 저장
            window_info = {
                "window_index": window_index,
                "embedding": embedding,
                "all_transcripts": transcripts,
                "first_sentence": first_sentence,
                "last_sentence": last_sentence,
                "first_timestamp": first_timestamp,
                "start_index": start_index,  # DataFrame 인덱스
                "end_index": end_index,  # DataFrame 인덱스
            }

            processed_windows.append(window_info)

        logger.info(f"총 {len(processed_windows)}개의 윈도우를 생성했습니다.")
        return processed_windows

    except Exception as e:
        logger.error(f"윈도우 생성 중 오류 발생: {str(e)}")
        raise


def calculate_window_similarities(processed_windows):
    """
    인접한 윈도우 간의 코사인 유사도를 계산합니다.

    Args:
        processed_windows (list): 처리된 윈도우 정보 리스트

    Returns:
        list: 경계 점수 결과를 담은 딕셔너리 리스트
    """
    try:
        logger.info("윈도우 간 코사인 유사도 계산 중...")
        boundary_results = []

        for i in range(len(processed_windows) - 1):
            win_i = processed_windows[i]
            win_i1 = processed_windows[i + 1]

            # 임베딩을 numpy array로 변환하고 reshape
            embedding_i = np.array(win_i["embedding"]).reshape(1, -1)
            embedding_i1 = np.array(win_i1["embedding"]).reshape(1, -1)

            # 코사인 유사도 계산
            similarity = cosine_similarity(embedding_i, embedding_i1)[0][0]

            result = {
                "score": float(similarity),
                "window_N": i,
                "window_N1": i + 1,
            }

            boundary_results.append(result)

        logger.info(f"총 {len(boundary_results)}개의 경계 점수를 계산했습니다.")
        return boundary_results

    except Exception as e:
        logger.error(f"점수 계산 중 오류 발생: {str(e)}")
        raise


def select_candidates(boundary_results):
    """
    Candidate window pairs를 선택합니다.
    - Similarity < 0.4인 쌍 선택
    - 없으면 Top 5 선택

    Args:
        boundary_results (list): 경계 점수 결과 리스트

    Returns:
        list: 선택된 candidate pairs
        bool: threshold 사용 여부
    """
    # Similarity < 0.4인 쌍 찾기
    threshold_candidates = [
        r for r in boundary_results if r["score"] < SIMILARITY_THRESHOLD
    ]

    if len(threshold_candidates) > 0:
        logger.info(
            f"Threshold 기준 ({SIMILARITY_THRESHOLD})으로 {len(threshold_candidates)}개의 candidate 선택"
        )
        return threshold_candidates, True
    else:
        # Top 5 선택
        sorted_results = sorted(boundary_results, key=lambda x: x["score"])
        top_n = sorted_results[:TOP_N_CANDIDATES]
        logger.info(f"Threshold 기준으로 candidate 없음. Top {TOP_N_CANDIDATES} 선택")
        return top_n, False


def fine_grained_refinement(df, candidate_pair, processed_windows, k, client):
    """
    Fine-grained refinement: 각 candidate pair에 대해 sentence-level similarity 계산.

    Args:
        df (pd.DataFrame): 전체 문장 DataFrame
        candidate_pair (dict): Candidate window pair 정보
        processed_windows (list): 처리된 윈도우 정보 리스트
        k (int): 윈도우 크기
        client (OpenAI): OpenAI 클라이언트

    Returns:
        int: Global cut point (sentence index)
    """
    win_N_idx = candidate_pair["window_N"]
    win_N1_idx = candidate_pair["window_N1"]

    win_N = processed_windows[win_N_idx]
    win_N1 = processed_windows[win_N1_idx]

    # Window i의 시작부터 Window i+1의 끝까지 2k sentences 추출
    start_idx = win_N["start_index"]
    end_idx = win_N1["end_index"]

    logger.info(
        f"  Candidate {win_N_idx}->{win_N1_idx}: 인덱스 {start_idx}부터 {end_idx}까지 처리"
    )

    segment_df = df.iloc[start_idx:end_idx].copy()
    segment_df = segment_df.reset_index(drop=True)

    if len(segment_df) < 2:
        # 문장이 2개 미만이면 첫 번째 문장 다음이 cut point
        return start_idx + 1

    # 각 sentence에 대한 embedding 생성
    logger.info(f"    {len(segment_df)}개 문장의 embedding 생성 중...")
    sentence_embeddings = []
    for idx, row in segment_df.iterrows():
        text = row["text"]
        embedding = get_embedding(text, client)
        sentence_embeddings.append(embedding)

    # Adjacent sentence similarity 계산
    similarities = []
    for i in range(len(sentence_embeddings) - 1):
        emb_i = np.array(sentence_embeddings[i]).reshape(1, -1)
        emb_i1 = np.array(sentence_embeddings[i + 1]).reshape(1, -1)
        similarity = cosine_similarity(emb_i, emb_i1)[0][0]
        similarities.append((i, float(similarity)))

    # Minimum similarity 지점 찾기
    min_sim_idx = min(similarities, key=lambda x: x[1])[0]
    global_cut_point = start_idx + min_sim_idx + 1  # 다음 문장의 인덱스

    logger.info(
        f"    최소 유사도 지점: 문장 {min_sim_idx}와 {min_sim_idx+1} 사이 (유사도: {similarities[min_sim_idx][1]:.6f})"
    )

    return global_cut_point


def construct_segments(df, cut_points, client):
    """
    Cut points를 기반으로 segments 생성 + 각 segment embedding 저장
    """

    logger.info(f"Cut points 기반으로 {len(cut_points)}개의 segments 생성 중...")

    segments = []
    cut_points = sorted(cut_points)
    cut_points = [0] + cut_points + [len(df)]

    for seg_num in range(len(cut_points) - 1):
        start_idx = cut_points[seg_num]
        end_idx = cut_points[seg_num + 1]

        segment_df = df.iloc[start_idx:end_idx]
        if len(segment_df) == 0:
            continue

        # Segment 텍스트 만들기
        segment_text = " ".join(segment_df["text"].tolist())
        start_timestamp = float(segment_df.iloc[0]["start_timestamp"])

        # ⭐ segment-level embedding 생성
        try:
            segment_embedding = get_embedding(segment_text, client)
        except Exception:
            segment_embedding = None

        segment_info = {
            "seg_num": seg_num + 1,
            "start_timestamp": start_timestamp,
            "text": segment_text,
            "embedding": segment_embedding,   # ⭐ 추가된 필드
        }

        segments.append(segment_info)

    logger.info(f"총 {len(segments)}개의 segments 생성 완료 (+ embedding 저장)")
    return segments



def save_results(segments, input_filepath, k, used_threshold, output_dir):
    """
    결과를 JSON 파일로 저장합니다.

    Args:
        segments (list): Segment 정보 리스트
        input_filepath (str): 입력 파일 경로
        k (int): 윈도우 크기
        used_threshold (bool): Threshold 사용 여부
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        input_base = os.path.splitext(os.path.basename(input_filepath))[0]
        output_file = os.path.join(
            output_dir,
            f"segment_result_{input_base}_k_{k}_thr_{SIMILARITY_THRESHOLD}.json",
        )

        output_data = {
            "k": k,
            "threshold": used_threshold,
            "segments": segments,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"결과 저장 완료: {output_file}")
        logger.info(f"총 {len(segments)}개의 segments 저장됨")

    except Exception as e:
        logger.error(f"결과 저장 중 오류 발생: {str(e)}")
        raise


def main(filepath, k, output_dir):
    """메인 함수: 전체 프로세스를 오케스트레이션합니다."""
    # 명령줄 인수 파싱
    parser = argparse.ArgumentParser(
        description="Hierarchical Video Segmentation Script"
    )
    # parser.add_argument(
    #     "--filepath",
    #     type=str,
    #     required=True,
    #     help="분석할 JSON 파일의 경로 (예: transcripts/4PFhxLjw5vw.json)",
    # )
    # parser.add_argument(
    #     "--k",
    #     type=int,
    #     default=5,
    #     help="사용할 윈도우 크기 (기본값: 5)",
    # )

    args = parser.parse_args()
    args.filepath = filepath
    args.k = k


    try:

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)

        # Step 1: Coarse-grained Boundary Detection
        logger.info("\n" + "=" * 80)
        logger.info("[Step 1] Coarse-grained Boundary Detection (Window Level)")
        logger.info("=" * 80)

        df = load_data(args.filepath)
        processed_windows = create_windows(df, args.k, client)
        boundary_results = calculate_window_similarities(processed_windows)

        # Candidate selection
        candidates, used_threshold = select_candidates(boundary_results)
        logger.info(f"선택된 candidate pairs: {len(candidates)}개")

        # Step 2: Fine-grained Refinement
        logger.info("\n" + "=" * 80)
        logger.info("[Step 2] Fine-grained Refinement (Sentence Level)")
        logger.info("=" * 80)

        global_cut_points = []
        for idx, candidate in enumerate(candidates, 1):
            logger.info(f"\nCandidate {idx}/{len(candidates)} 처리 중...")
            cut_point = fine_grained_refinement(
                df, candidate, processed_windows, args.k, client
            )
            global_cut_points.append(cut_point)

        # 중복 제거 및 정렬
        global_cut_points = sorted(list(set(global_cut_points)))
        logger.info(f"\n총 {len(global_cut_points)}개의 unique cut points 발견")

        # Step 3: Construct Segments
        logger.info("\n" + "=" * 80)
        logger.info("[Step 3] Construct Segments")
        logger.info("=" * 80)

        segments = construct_segments(df, global_cut_points, client)

        # Save results
        save_results(segments, args.filepath, args.k, used_threshold, output_dir)

        logger.info("\n" + "=" * 80)
        logger.info("[SUCCESS] 모든 작업이 완료되었습니다!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\n작업 중 오류 발생: {str(e)}")
        import traceback

        traceback.print_exc()
        return


if __name__ == "__main__":
    # SOURCE_DIR = "transcripts"
    SOURCE_DIR = "preprocessed_transcripts"
    output_dir = "Stage1_Segmentation/segment_result"

    for file in os.listdir(SOURCE_DIR):
        if file.endswith(".json"):
            print(f"[INFO] 세그먼트 찾기 중: {file}")
            main(os.path.join(SOURCE_DIR, file), K, output_dir)
            print(f"[SUCCESS] 세그먼트 찾기 완료: {file}")
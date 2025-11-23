from typing import Dict, List
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


EMBED_MODEL = "text-embedding-3-small"

def embed_text(t):
    if not t.strip():
        return None
    try:
        r = client.embeddings.create(model=EMBED_MODEL, input=t)
        return np.array(r.data[0].embedding, dtype=float)
    except:
        return None

# 비교대상 1) Stage 1: ours + Stage 2: LLM
def summarize_segment_llm(segments: List[Dict], model_name: str = "gpt-4o-mini") -> List[Dict]:
    ans = []

    for segment in segments:
        out = {
            "seg_num": segment["seg_num"],
            "start_timestamp": segment["start_timestamp"]
        }

        text = segment["text"].strip()

        prompt = f"""
                You are an AI assistant that generates short YouTube chapter titles.

                Instructions:
                - Read the text below.
                - Produce a short, clean, meaningful title.
                - **Only output the title.**
                - Do NOT output JSON.
                - Do NOT output quotes.
                - Do NOT use code blocks.
                - Do NOT mention "summary", "segment", "chapter", or timestamps.
                - Just output raw text.

                Text:
                {text}
                """

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system",
                 "content": "You generate short, clean, human-like YouTube chapter titles. Output must be plain text only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3
        )

        # GPT가 의도치 않게 quote 붙일 경우 제거
        title = response.choices[0].message.content.strip()
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1].strip()
        if title.startswith("'") and title.endswith("'"):
            title = title[1:-1].strip()
        if title.startswith("```"):
            title = title.replace("```", "").strip()
        if "{ " in title or "\"title\"" in title:
            # JSON 비슷하게 출력한 경우 → 내부 title만 뽑기
            import re
            m = re.search(r'"title"\s*:\s*"([^"]+)"', title)
            if m:
                title = m.group(1)

        out["title"] = title
        ans.append(out)

    return ans


# 비교대상 2) End-to-end LLM (Stage 1 & 2: llm) 
def summarize_all_llm(sentences: List[Dict], model_name: str = "gpt-4o-mini") -> List[Dict]:
    """
    Sentences: List of dict objects like:
      {"start_timestamp": float, "text": str}
    """

    sentences = format_transcript_with_timestamps(sentences)

    """구간 분할 및 소제목 생성을 위한 프롬프트"""
    prompt = f"""The following is an English YouTube video transcript with timestamp information.

                Transcript:
                {sentences}

                **Task:**
                1. Divide this transcript into multiple segments based on natural changes in topics or content.
                2. Accurately identify the start time (start_timestamp) of each segment and generate a concise and clear title that represents that segment.

                **Segment Division Criteria:**
                - Changes in topics or themes
                - Natural flow of conversation or explanation
                - Important content transition points
                - Appropriate segment length (not too short or too long)

                **Subtitle Writing Guidelines:**
                - Summarize the core content of each segment concisely (approximately 10-20 words)
                - Use clear and easy-to-understand expressions
                - Avoid unnecessary modifiers or decorative expressions
                - Prefer specific and informative titles

                **Output Format:**
                You must respond only in the following JSON object format. Do not include any other explanations or text.

                {{
                "segments": [
                    {{
                    "start_timestamp": 0.12,
                    "title": "title 1"
                    }},
                    {{
                    "start_timestamp": 103.5,
                    "title": "title 2"
                    }},
                    {{
                    "start_timestamp": 287.2,
                    "title": "title 3"
                    }}
                ]
                }}

                **Important Notes:**
                - You must respond in valid JSON object format ({{"segments": [...]}})
                - Use the actual timestamp values from the transcript for start_timestamp
                - The first segment should always start at 0 seconds or the timestamp of the first utterance
                - Return only the JSON object without any other text
                """

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at analyzing video transcripts, dividing them into segments, and generating subtitles. Always respond only in valid JSON format.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,  # 일관성 있는 결과를 위해 낮은 temperature
        response_format={"type": "json_object"},  # JSON 형식 강제
    )

    response_text = response.choices[0].message.content
    try:
        parsed_response = json.loads(response_text)
        baseline_data = parsed_response["segments"]
    except json.JSONDecodeError:
        baseline_data = parse_json_response(response_text)

    validated_data = []
    for item in baseline_data:
        if (isinstance(item, dict) and "start_timestamp" in item and "title" in item):
            validated_data.append(
                {
                    "start_timestamp": float(item["start_timestamp"]),
                    "title": str(item["title"]).strip(),
                    "embedding": embed_text(str(item["title"]).strip()).tolist()
                }
            )
    
    validated_data.sort(key=lambda x: x["start_timestamp"])

    return validated_data


def parse_json_response(response_text):
    """GPT 응답에서 JSON 추출"""
    # JSON 코드 블록이 있는 경우 추출
    json_match = re.search(r"```json\s*(\[.*?\])\s*```", response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(1)
    else:
        # 코드 블록 없이 JSON만 있는 경우
        json_match = re.search(r"(\[.*?\])", response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)

    # JSON 파싱 시도
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"  [경고] JSON 파싱 실패: {str(e)}")
        print(f"  [경고] 응답 텍스트 (처음 500자): {response_text[:500]}")
        return None


def format_transcript_with_timestamps(transcript_data):
    """트랜스크립트를 시간 정보와 함께 텍스트로 포맷팅"""
    formatted_lines = []
    for item in transcript_data:
        timestamp = item.get("start_timestamp", 0)
        transcript = item.get("text", "").strip()
        if transcript:
            # 시간을 분:초 형식으로 변환
            minutes = int(timestamp // 60)
            seconds = timestamp % 60
            time_str = f"{minutes:02d}:{seconds:05.2f}"
            formatted_lines.append(f"[{time_str}] {transcript}")
    return "\n".join(formatted_lines)


if __name__ == "__main__":
    SOURCE_DIR = "Stage1_Segmentation/segment_result"
    TRANSCRIPT_DIR = "preprocessed_transcripts"
    OUTPUT_DIR = "Stage2_Summarization/summarized_results"
    LLM_OUTPUT_DIR = "Stage2_Summarization/llm_all_summarized_results"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for file in os.listdir(SOURCE_DIR):
        print(f"Processing {file}...")
        if file.endswith(".json"):
            with open(os.path.join(SOURCE_DIR, file), "r", encoding="utf-8") as f:
                data = json.load(f)

            segments = data["segments"]

            llm_stage_2_summarized_segments = data.copy()
            llm_stage_2_summarized_segments["segments"] = summarize_segment_llm(segments)

            name = file.split('.json')[0].split('segment_result_preprocessed_')[1]
            path = os.path.join(OUTPUT_DIR, name)
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "llm_stage_2_summarized_segments.json"), "w", encoding="utf-8") as f:
                json.dump(llm_stage_2_summarized_segments, f, ensure_ascii=False, indent=2)

    os.makedirs(LLM_OUTPUT_DIR, exist_ok=True)

    for file in os.listdir(TRANSCRIPT_DIR):
        if file.endswith('.json'):
            with open(os.path.join(TRANSCRIPT_DIR, file), "r", encoding="utf-8") as f:
                data = json.load(f)

            segments = data["sentences"]

            llm_stage_all_summarized_segments = data.copy()
            llm_stage_all_summarized_segments["sentences"] = summarize_all_llm(segments)

            name = file.split('preprocessed_')[1]

            path = os.path.join(LLM_OUTPUT_DIR, name)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(llm_stage_all_summarized_segments, f, ensure_ascii=False, indent=2)

import os
import json
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import CouldNotRetrieveTranscript, NoTranscriptFound

import re
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

# 상수 정의
INPUT_FILE = "video_id.txt"
OUTPUT_DIR = "transcripts"

# 출력 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_transcripts(video_id):
    try:
        # API 호출: 한국어 트랜스크립트 가져오기
        fetched_transcript = api.fetch(video_id, languages=["en"])

        # 데이터 변환: JSON 포맷으로 변환
        transcript_data = {"all_text": None, "sentences": []}
        base_txt = ''

        all_txt = ''

        snippets = fetched_transcript.snippets
        base_start_time = snippets[0].start

        for snippet in snippets:

            start_time = snippet.start
            transcript = snippet.text

            all_txt = all_txt + transcript + " "

            if any(c in transcript for c in ['.', '?', '!']):
                # ., ?, ! 기준으로 문장 구분
                sentences = [s.strip() for s in re.split(r'[.!?]', transcript)]
                sentences[0] = base_txt + " " + sentences[0].strip()
                for i, sentence in enumerate(sentences[:-1]):

                    time_tmp = start_time

                    if base_txt != '':
                        time_tmp = base_start_time
                        base_txt = ''
                        
                    transcript_data["sentences"].append(
                        {
                            "start_timestamp": time_tmp,
                            "text": sentence,
                        }
                    )
                base_txt = sentences[-1]
                base_start_time = start_time

            else:
                if base_txt != '':
                    base_txt = base_txt.strip() + " "
                else:
                    base_start_time = start_time

                base_txt = base_txt + transcript.strip()

        transcript_data["all_text"] = all_txt

        return transcript_data

    except Exception as e:
        print(f"[ERROR] {video_id}: 예상치 못한 오류가 발생했습니다. ({str(e)})")


def parse_timestamp_to_seconds(t: str) -> int:
    """ HH:MM:SS 또는 MM:SS 을 초 단위로 변환 """
    parts = t.split(":")
    parts = list(map(int, parts))

    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m, s = parts
        return m * 60 + s
    else:
        return 0  # fallback


def fetch_gt(video_id):
    try:
        API_KEY = os.getenv("GOOGLE_API_KEY")
        youtube = build("youtube", "v3", developerKey=API_KEY)

        response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()

        description = response["items"][0]["snippet"]["description"]

        gt = []

        # --- Case 1: 시간 먼저 ---
        pattern_time_first = re.compile(
            r"^(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+)",
            re.MULTILINE
        )

        # --- Case 2: 제목 먼저 ---
        pattern_title_first = re.compile(
            r"^(.+?)\s+(\d{1,2}:\d{2}(?::\d{2})?)$",
            re.MULTILINE
        )

        # Case 1 처리
        for timestamp, title in pattern_time_first.findall(description):
            gt.append({
                "start_timestamp": parse_timestamp_to_seconds(timestamp),
                "text": title.strip()
            })

        # Case 2 처리
        for title, timestamp in pattern_title_first.findall(description):
            gt.append({
                "start_timestamp": parse_timestamp_to_seconds(timestamp),
                "text": title.strip()
            })

        # 시간순 정렬
        gt.sort(key=lambda x: x["start_timestamp"])

        return gt

    except Exception as e:
        print(f"[ERROR] {video_id}: {str(e)}")
        return []


# 입력 파일 읽기
video_ids = []
try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            video_id = line.strip()  # 공백과 개행 문자 제거
            if video_id:  # 비어있는 줄 건너뛰기
                video_ids.append(video_id)
except FileNotFoundError:
    print(f"[ERROR] {INPUT_FILE} 파일을 찾을 수 없습니다.")
    exit(1)

print(f"총 {len(video_ids)}개의 비디오 ID를 읽었습니다.\n")

# 트랜스크립트 처리 루프
api = YouTubeTranscriptApi()
for video_id in video_ids:
    data = fetch_transcripts(video_id)
    print(f"[SUCCESS] {video_id} 트랜스크립트 처리 완료.")
    data["gt"] = fetch_gt(video_id)

    output_file = os.path.join(OUTPUT_DIR, f"{video_id}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
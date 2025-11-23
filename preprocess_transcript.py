#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_transcript.py — 완전 개선된 버전 (NER 기반 고유명사 보호)
 - 비언어 표현 제거
 - 구두점 제거
 - NER 기반 고유명사 보호
 - Spelling correction (고유명사 제외)
 - Stopword 제거
 - Lemmatization (고유명사 제외)
"""

import argparse
import json
import os
import re
from typing import List, Dict, Any

# -----------------------------
# NLTK
# -----------------------------
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# -----------------------------
# TextBlob (spelling correction)
# -----------------------------
from textblob import TextBlob

# -----------------------------
# spaCy NER
# -----------------------------
import spacy

# -----------------------------
# contractions (contraction expansion)
# -----------------------------
import contractions


# -----------------------------------------------------------------------------
# 전역 설정
# -----------------------------------------------------------------------------
NONVERBAL_PATTERN = re.compile(r"\[[^\]]*\]")
PUNCT_PATTERN = re.compile(r"[^\w\s]")

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# spaCy NER 로드
import en_core_web_sm
NER = en_core_web_sm.load()


# -----------------------------------------------------------------------------
# 전처리 함수
# -----------------------------------------------------------------------------


def merge_same_timestamp(sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """start_timestamp가 동일한 문장을 병합합니다."""
    merged = []
    current_timestamp = None
    buffer = []

    for item in sentences:
        ts = item.get("start_timestamp")
        text = item.get("text", "").strip()

        if current_timestamp is None:
            current_timestamp = ts
            buffer.append(text)
            continue

        if ts == current_timestamp:
            buffer.append(text)
        else:
            merged.append(
                {"start_timestamp": current_timestamp, "text": " ".join(buffer)}
            )
            current_timestamp, buffer = ts, [text]

    if buffer:
        merged.append(
            {"start_timestamp": current_timestamp, "text": " ".join(buffer)}
        )

    return merged


def remove_nonverbal(text: str) -> str:
    """대괄호 비언어 표현 + 구두점 제거."""
    text = NONVERBAL_PATTERN.sub(" ", text)
    text = text.replace(">>", " ")
    text = PUNCT_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def expand_contractions(text):
    return contractions.fix(text)


# -----------------------------------------------------------------------------
# NER 기반 spelling correction 보호
# -----------------------------------------------------------------------------
def correct_spelling_protected(text: str) -> str:
    """
    TextBlob spelling correction을 적용하되,
    PERSON/ORG/GPE/PRODUCT 등 고유명사는 그대로 보존한다.
    """
    if not text:
        return text

    doc = NER(text)
    PROTECT_TYPES = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT"}

    corrected_tokens = []

    for token in doc:
        if token.ent_type_ in PROTECT_TYPES:
            corrected_tokens.append(token.text)
        else:
            try:
                corrected = str(TextBlob(token.text).correct())
                corrected_tokens.append(corrected)
            except Exception:
                corrected_tokens.append(token.text)

    return " ".join(corrected_tokens)


# -----------------------------------------------------------------------------
# POS → WordNet POS 변환
# -----------------------------------------------------------------------------
def get_wordnet_pos(treebank_tag: str):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return None


# -----------------------------------------------------------------------------
# NER 기반 lemma + stopword 보호
# -----------------------------------------------------------------------------
def lemmatize_and_remove_stopwords_protected(text: str, stopword_removal: bool = False) -> str:
    """
    stopword 제거 + lemmatization을 수행하되,
    NER 고유명사는 보호하여 원형 그대로 유지.
    """

    if not text:
        return text

    doc = NER(text)
    PROTECT_TYPES = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT"}

    protected_indices = set()
    for ent in doc.ents:
        if ent.label_ in PROTECT_TYPES:
            for i in range(ent.start, ent.end):
                protected_indices.add(i)

    tokens = nltk.word_tokenize(text)
    tagged_tokens = pos_tag(tokens)

    cleaned_tokens = []

    for idx, (token, tag) in enumerate(tagged_tokens):
        # 고유명사 스팬 보호
        if idx in protected_indices:
            cleaned_tokens.append(token)
            continue

        lower = token.lower().strip()

        # stopwords 제거
        if stopword_removal:
            
            STOPWORD_WHITELIST = {
                    # negations
                    "not", "no", "nor", "never", "without", "hardly", "barely",

                    # modal verbs
                    "will", "would", "should", "could", "can", "may", "might", "must",
                    "do", "did", "does",

                    # discourse markers / transitions
                    "but", "however", "although", "though",
                    "so", "now", "then", "right", "well", "okay", "anyway",
                    "actually", "basically", "first", "second", "next", "finally", "overall",
                    
                    # conditional
                    "if", "unless", "except"
                }
            
            if lower in STOPWORDS and lower not in STOPWORD_WHITELIST:
                continue

        # lemma
        wn_pos = get_wordnet_pos(tag)
        lemma = (
            LEMMATIZER.lemmatize(lower, wn_pos)
            if wn_pos
            else LEMMATIZER.lemmatize(lower)
        )
        cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)


# -----------------------------------------------------------------------------
# 전체 파이프라인
# -----------------------------------------------------------------------------
def preprocess_sentence(text: str, spelling_correction: bool = False, stopword_removal: bool = False) -> str:

    if not text:
        return ""

    # contraction expansion
    text = expand_contractions(text)

    # remove nonverbal
    text = remove_nonverbal(text)

    # spelling correction (NER 보호)
    if spelling_correction:
        text = correct_spelling_protected(text)

    # lemma + stopword (NER 보호)
    text = lemmatize_and_remove_stopwords_protected(text, stopword_removal=stopword_removal)

    # 공백 정리
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------------------------------------------------------
# 파일 처리
# -----------------------------------------------------------------------------
def process_file(filepath: str, output_dir: str, key='segments', spelling_correction: bool = False, stopword_removal: bool = False) -> None:
    print(f"[INFO] 입력 파일 로드: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get(key, [])
    if not segments:
        raise ValueError(f"{key} 데이터가 비어있습니다.")

    segments = merge_same_timestamp(segments)

    processed = []

    print("[INFO] 전처리 시작 (NER 보호)")

    for seg in segments:
        clean = preprocess_sentence(seg.get("text", ""), spelling_correction=spelling_correction, stopword_removal=stopword_removal)
        if not clean:
            continue

        tmp = seg.copy()
        tmp['text'] = clean

        processed.append(tmp)

    data[key] = processed

    # data = {k: v for k, v in data.items() if k != "gt"}

    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(filepath))[0]
    output_path = os.path.join(output_dir, f"preprocessed_{base}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] 저장 완료: {output_path}")


# -----------------------------------------------------------------------------
# 자동 처리 루프
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    SOURCE_DIR = "transcripts"
    OUTPUT_DIR = "preprocessed_transcripts"
    KEY = 'sentences'

    for file in os.listdir(SOURCE_DIR):
        if file.endswith(".json"):
            print(f"[INFO] 처리 중: {file}")
            process_file(
                os.path.join(SOURCE_DIR, file),
                output_dir=OUTPUT_DIR,
                key=KEY,
                spelling_correction=False,
                stopword_removal=True
            )
            print(f"[DONE] {file}")
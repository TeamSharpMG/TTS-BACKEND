# src/services/correction_service.py
"""Post-transcription spelling correction for Malagasy STT output.

Uses baiboly-words.csv as the authoritative vocabulary (candidate pool) and
baiboly.csv (the full Bible text) as a corpus to derive word/bigram
frequencies for scoring candidates — a noisy-channel spell corrector, not a
generic sampler: exact nearest-neighbour search in the lexicon, weighted by
how plausible the candidate is in context.
"""
import csv
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BAIBOLY_TEXT_CSV = BASE_DIR / "baiboly.csv"
BAIBOLY_WORDS_CSV = BASE_DIR / "baiboly-words.csv"
CACHE_PATH = BASE_DIR / "ai" / "stt" / "lexicon_cache.pkl"

TOKEN_RE = re.compile(r"[a-zàâäéèêëïîôöùûü'\-]+", re.IGNORECASE)
MAX_EDIT_DISTANCE = 2

_resources = None


def _tokenize(text: str) -> list:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def _build_resources() -> dict:
    word_freq = Counter()
    bigram_freq = Counter()

    with open(BAIBOLY_TEXT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tokens = _tokenize(row["text"])
            word_freq.update(tokens)
            bigram_freq.update(zip(tokens, tokens[1:]))

    vocab = set()
    with open(BAIBOLY_WORDS_CSV, encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                vocab.add(w)

    by_length = defaultdict(list)
    for w in vocab:
        by_length[len(w)].append(w)

    resources = {
        "word_freq": word_freq,
        "bigram_freq": bigram_freq,
        "vocab": vocab,
        "by_length": dict(by_length),
    }

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(resources, f)

    return resources


def _load_resources() -> dict:
    global _resources
    if _resources is not None:
        return _resources

    fresh_cache = (
        CACHE_PATH.exists()
        and CACHE_PATH.stat().st_mtime > BAIBOLY_TEXT_CSV.stat().st_mtime
        and CACHE_PATH.stat().st_mtime > BAIBOLY_WORDS_CSV.stat().st_mtime
    )
    if fresh_cache:
        with open(CACHE_PATH, "rb") as f:
            _resources = pickle.load(f)
    else:
        _resources = _build_resources()

    return _resources


def _levenshtein(a: str, b: str, max_dist: int) -> int:
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        row_min = cur[0]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            row_min = min(row_min, cur[j])
        if row_min > max_dist:
            return max_dist + 1
        prev = cur
    return prev[-1]


def _candidates(word: str, resources: dict, max_dist: int) -> list:
    by_length = resources["by_length"]
    pool = []
    for length in range(len(word) - max_dist, len(word) + max_dist + 1):
        pool.extend(by_length.get(length, []))
    return [w for w in pool if _levenshtein(word, w, max_dist) <= max_dist]


def _best_candidate(word: str, prev_word, resources: dict) -> str:
    max_dist = 1 if len(word) <= 4 else MAX_EDIT_DISTANCE
    candidates = _candidates(word, resources, max_dist)
    if not candidates:
        return word

    word_freq = resources["word_freq"]
    bigram_freq = resources["bigram_freq"]

    def score(cand: str) -> float:
        dist = _levenshtein(word, cand, max_dist)
        unigram_score = math.log(word_freq.get(cand, 0) + 1)
        bigram_score = 0.0
        if prev_word is not None:
            bigram_score = math.log(bigram_freq.get((prev_word, cand), 0) + 1)
        return -2.0 * dist + unigram_score + 1.5 * bigram_score

    return max(candidates, key=score)


def correct_text(text: str) -> str:
    resources = _load_resources()
    vocab = resources["vocab"]

    tokens = text.split()
    corrected = []
    prev_clean = None

    for tok in tokens:
        match = TOKEN_RE.findall(tok.lower())
        cleaned = match[0] if match else ""
        if not cleaned:
            corrected.append(tok)
            continue

        fixed = cleaned if cleaned in vocab else _best_candidate(cleaned, prev_clean, resources)
        corrected.append(fixed)
        prev_clean = fixed

    return " ".join(corrected)

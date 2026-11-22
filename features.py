"""
Phase 2: Context-Grounding Feature Extraction & Data Contract Validation.

Provides a scikit-learn compatible transformer (`ContextGroundingFeatureExtractor`)
that computes length-invariant and sentence-style-invariant lexical, entity,
numerical, sentence-support, and n-gram grounding features comparing a candidate
`answer` against the reference `context` and `question`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "doing", "done", "for", "from",
    "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "just", "may", "might",
    "more", "most", "must", "my", "no", "nor", "not", "now", "of", "on", "once",
    "only", "or", "other", "our", "ours", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "would",
    "you", "your", "yours", "also", "s", "t", "d", "m", "re", "ve", "ll",
}

NEGATION_WORDS: Set[str] = {
    "no", "not", "never", "neither", "none", "nobody", "nothing", "nowhere",
    "hardly", "scarcely", "barely", "doesnt", "dont", "didnt", "isnt", "arent",
    "wasnt", "werent", "hasnt", "havent", "hadnt", "wont", "wouldnt", "cant",
    "cannot", "couldnt", "shouldnt",
}

CONTRADICTION_CUES: Set[str] = {
    "actually", "however", "although", "instead", "contrary", "rather",
    "incorrect", "false", "wrong", "neither", "nor", "not", "unlikely",
}

WORD_RE = re.compile(r"[a-z0-9]+")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
CAP_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z0-9\-]{1,}\b")
SENT_SPLIT_RE = re.compile(r"[.!?]+")


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenization."""
    if not text:
        return []
    return WORD_RE.findall(text.lower())


def char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from normalized text."""
    cleaned = re.sub(r"\s+", " ", text.lower()).strip()
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def stem_simple(token: str) -> str:
    """Lightweight suffix normalization for morphological variants."""
    for suffix in ("ingly", "edly", "ment", "ness", "tion", "sion", "ing", "ers", "ies", "ed", "ly", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            if suffix == "ies":
                return token[:-3] + "y"
            return token[: -len(suffix)]
    return token


def extract_grounding_signals(context: str, question: str, answer: str) -> Dict[str, float]:
    """
    Compute interpretable, length-invariant and sentence-style-invariant grounding
    features for a single (context, question, answer) triplet.

    Words echoed from `question` are treated as question framing so that full-sentence
    grounded answers ("The Oberoi Group head office is in Delhi.") and short extractive
    answers ("Delhi") produce consistent grounding representations.
    """
    context_str = str(context or "").strip()
    question_str = str(question or "").strip()
    answer_str = str(answer or "").strip()

    ctx_tokens = tokenize(context_str)
    q_tokens = tokenize(question_str)
    ans_tokens = tokenize(answer_str)

    ctx_token_set = set(ctx_tokens)
    q_token_set = set(q_tokens)
    ans_token_set = set(ans_tokens)

    ctx_stems = {stem_simple(t) for t in ctx_token_set}
    q_stems = {stem_simple(t) for t in q_token_set}

    # 1. Coverage of answer tokens against (context U question)
    ans_content_tokens = [t for t in ans_tokens if t not in STOPWORDS]
    if not ans_content_tokens:
        ans_content_tokens = list(ans_tokens)

    if ans_tokens:
        in_ctx_or_q_count = sum(
            1
            for t in ans_tokens
            if t in ctx_token_set
            or t in q_token_set
            or t in STOPWORDS
            or stem_simple(t) in ctx_stems
            or stem_simple(t) in q_stems
        )
        ans_ctx_or_q_overlap = in_ctx_or_q_count / len(ans_tokens)
    else:
        ans_ctx_or_q_overlap = 0.0

    if ans_content_tokens:
        content_in_ctx_or_q = sum(
            1
            for t in ans_content_tokens
            if t in ctx_token_set
            or t in q_token_set
            or stem_simple(t) in ctx_stems
            or stem_simple(t) in q_stems
        )
        content_ctx_or_q_overlap = content_in_ctx_or_q / len(ans_content_tokens)

        content_in_ctx = sum(
            1 for t in ans_content_tokens if t in ctx_token_set or stem_simple(t) in ctx_stems
        )
        content_ctx_overlap = content_in_ctx / len(ans_content_tokens)
    else:
        content_ctx_or_q_overlap = 0.0
        content_ctx_overlap = 0.0

    # 2. New-information tokens in answer (content tokens NOT already in the question)
    new_info_tokens = [
        t for t in ans_content_tokens if t not in q_token_set and stem_simple(t) not in q_stems
    ]
    eval_tokens = new_info_tokens if new_info_tokens else ans_content_tokens

    if eval_tokens:
        eval_in_ctx = sum(
            1 for t in eval_tokens if t in ctx_token_set or stem_simple(t) in ctx_stems
        )
        new_info_grounding_ratio = eval_in_ctx / len(eval_tokens)
        novel_ungrounded_token_ratio = 1.0 - new_info_grounding_ratio
        ungrounded_new_token_count = float(len(eval_tokens) - eval_in_ctx)
    else:
        new_info_grounding_ratio = 0.0
        novel_ungrounded_token_ratio = 1.0
        ungrounded_new_token_count = 1.0

    # 3. Contiguous phrase & bigram grounding of the core answer span (`eval_tokens`)
    norm_ctx = " ".join(ctx_tokens)
    ctx_content = [t for t in ctx_tokens if t not in STOPWORDS]
    norm_ctx_content = " ".join(ctx_content)
    eval_span = " ".join(eval_tokens)

    new_info_substring_in_ctx = (
        1.0
        if (eval_span and (eval_span in norm_ctx or eval_span in norm_ctx_content))
        else 0.0
    )

    if len(eval_tokens) >= 2:
        ctx_bigrams = set(zip(ctx_tokens, ctx_tokens[1:]))
        ctx_content_bigrams = set(zip(ctx_content, ctx_content[1:]))
        target_bigrams = list(zip(eval_tokens, eval_tokens[1:]))
        bg_hits = sum(
            1 for bg in target_bigrams if bg in ctx_bigrams or bg in ctx_content_bigrams
        )
        bigram_grounding_ratio = bg_hits / len(target_bigrams)
    else:
        bigram_grounding_ratio = new_info_substring_in_ctx

    # 4. Best supporting sentence overlap in context (for `eval_tokens` + `ans_content_tokens`)
    ctx_sentences = [s.strip() for s in SENT_SPLIT_RE.split(context_str) if s.strip()]
    best_sentence_support = 0.0
    best_sentence_full_content_support = 0.0
    if ctx_sentences and eval_tokens:
        for sent in ctx_sentences:
            s_toks = set(tokenize(sent))
            s_stems = {stem_simple(t) for t in s_toks}
            hits = sum(1 for t in eval_tokens if t in s_toks or stem_simple(t) in s_stems)
            ratio = hits / len(eval_tokens)
            if ratio > best_sentence_support:
                best_sentence_support = ratio

            if ans_content_tokens:
                full_hits = sum(
                    1
                    for t in ans_content_tokens
                    if t in s_toks or stem_simple(t) in s_stems or t in q_token_set
                )
                full_ratio = full_hits / len(ans_content_tokens)
                if full_ratio > best_sentence_full_content_support:
                    best_sentence_full_content_support = full_ratio
    else:
        best_sentence_support = new_info_grounding_ratio
        best_sentence_full_content_support = content_ctx_or_q_overlap

    # 5. Character 3-gram containment of core answer span in context
    ans_char_ngrams = char_ngrams(eval_span, n=3)
    ctx_char_ngrams = char_ngrams(norm_ctx, n=3)
    if ans_char_ngrams:
        char_ngram_containment = len(ans_char_ngrams & ctx_char_ngrams) / len(ans_char_ngrams)
    else:
        char_ngram_containment = 0.0

    # 6. Numerical / Year / Quantity grounding
    ctx_numbers = set(NUMBER_RE.findall(context_str))
    q_numbers = set(NUMBER_RE.findall(question_str))
    ans_numbers = NUMBER_RE.findall(answer_str)
    new_ans_numbers = [n for n in ans_numbers if n not in q_numbers]
    if new_ans_numbers:
        grounded_nums = sum(1 for n in new_ans_numbers if n in ctx_numbers)
        number_grounding_ratio = grounded_nums / len(new_ans_numbers)
        ungrounded_number_flag = 1.0 if grounded_nums < len(new_ans_numbers) else 0.0
    elif ans_numbers:
        grounded_nums = sum(1 for n in ans_numbers if n in ctx_numbers)
        number_grounding_ratio = grounded_nums / len(ans_numbers)
        ungrounded_number_flag = 1.0 if grounded_nums < len(ans_numbers) else 0.0
    else:
        number_grounding_ratio = 1.0
        ungrounded_number_flag = 0.0

    # 7. Capitalized Proper Noun / Named Entity grounding
    ctx_entities = {e.lower() for e in CAP_ENTITY_RE.findall(context_str)}
    q_entities = {e.lower() for e in CAP_ENTITY_RE.findall(question_str)}
    ans_entities = [e.lower() for e in CAP_ENTITY_RE.findall(answer_str)]
    new_ans_entities = [
        e for e in ans_entities if e not in q_entities and e not in STOPWORDS
    ]
    if new_ans_entities:
        grounded_ents = sum(
            1
            for e in new_ans_entities
            if e in ctx_entities or e in ctx_token_set or stem_simple(e) in ctx_stems
        )
        entity_grounding_ratio = grounded_ents / len(new_ans_entities)
        ungrounded_entity_flag = 1.0 if grounded_ents < len(new_ans_entities) else 0.0
    else:
        entity_grounding_ratio = 1.0
        ungrounded_entity_flag = 0.0

    # 8. Negation & Contradiction cues
    has_negation_in_ans = 1.0 if (ans_token_set & NEGATION_WORDS) else 0.0
    has_negation_in_ctx = 1.0 if (ctx_token_set & NEGATION_WORDS) else 0.0
    negation_mismatch_flag = 1.0 if (has_negation_in_ans == 1.0 and has_negation_in_ctx == 0.0) else 0.0
    contradiction_cue_flag = 1.0 if (ans_token_set & CONTRADICTION_CUES) else 0.0

    return {
        "ans_ctx_or_q_overlap": float(ans_ctx_or_q_overlap),
        "content_ctx_or_q_overlap": float(content_ctx_or_q_overlap),
        "content_ctx_overlap": float(content_ctx_overlap),
        "new_info_grounding_ratio": float(new_info_grounding_ratio),
        "novel_ungrounded_token_ratio": float(novel_ungrounded_token_ratio),
        "ungrounded_new_token_count": float(ungrounded_new_token_count),
        "new_info_substring_in_ctx": float(new_info_substring_in_ctx),
        "bigram_grounding_ratio": float(bigram_grounding_ratio),
        "best_sentence_support": float(best_sentence_support),
        "best_sentence_full_content_support": float(best_sentence_full_content_support),
        "char_ngram_containment": float(char_ngram_containment),
        "number_grounding_ratio": float(number_grounding_ratio),
        "ungrounded_number_flag": float(ungrounded_number_flag),
        "entity_grounding_ratio": float(entity_grounding_ratio),
        "ungrounded_entity_flag": float(ungrounded_entity_flag),
        "negation_mismatch_flag": float(negation_mismatch_flag),
        "contradiction_cue_flag": float(contradiction_cue_flag),
    }


FEATURE_NAMES: List[str] = list(
    extract_grounding_signals("Sample context.", "Sample question?", "Sample answer.").keys()
)


class ContextGroundingFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer that converts an iterable of (context, question, answer)
    records (dicts, DataFrames, or 3-tuples) into a numeric feature matrix of shape
    (n_samples, n_features).
    """

    def fit(self, X: Any, y: Any = None) -> "ContextGroundingFeatureExtractor":
        self.feature_names_in_ = FEATURE_NAMES
        return self

    def transform(self, X: Any) -> np.ndarray:
        rows: List[List[float]] = []
        for item in self._iter_records(X):
            signals = extract_grounding_signals(
                context=item[0],
                question=item[1],
                answer=item[2],
            )
            rows.append([signals[name] for name in FEATURE_NAMES])
        return np.asarray(rows, dtype=np.float64)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(FEATURE_NAMES, dtype=object)

    @staticmethod
    def _iter_records(X: Any) -> Iterable[Tuple[str, str, str]]:
        if hasattr(X, "to_dict"):
            for row in X.to_dict(orient="records"):
                yield (
                    str(row.get("context", row.get("knowledge", ""))),
                    str(row.get("question", "")),
                    str(row.get("answer", "")),
                )
        else:
            for item in X:
                if isinstance(item, dict):
                    yield (
                        str(item.get("context", item.get("knowledge", ""))),
                        str(item.get("question", "")),
                        str(item.get("answer", "")),
                    )
                else:
                    yield (str(item[0]), str(item[1]), str(item[2]))

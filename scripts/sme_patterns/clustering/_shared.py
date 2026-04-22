"""Shared clustering primitives.

Simple, explainable, rule-based + tf-idf + k-means. No black-box models.
Each cluster carries its member indexes and top tf-idf terms so the monthly
report can explain the cluster in one sentence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TextCluster:
    centroid_index: int
    member_indexes: list[int] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    vectorizer: TfidfVectorizer | None = None
    cluster_vector: np.ndarray | None = None


def choose_k(n_samples: int) -> int:
    """Pick a k that scales with sample size but stays interpretable.

    ``k = min(20, max(2, round(sqrt(n / 3))))`` for n >= 2.
    """
    if n_samples <= 1:
        return 1
    raw = round(math.sqrt(n_samples / 3.0))
    return max(2, min(20, raw))


def cluster_texts(texts: list[str], *, k: int | None = None) -> list[TextCluster]:
    """Cluster text inputs into k groups; preserves original index ordering."""
    if not texts:
        return []

    chosen_k = k or choose_k(len(texts))
    chosen_k = max(1, min(chosen_k, len(texts)))

    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[A-Za-z0-9_\-]{2,}\b",
        max_features=5000,
    )
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        # All inputs were stop-words-only; fall back to single cluster.
        return [
            TextCluster(
                centroid_index=0,
                member_indexes=list(range(len(texts))),
                terms=[],
            )
        ]

    if chosen_k == 1:
        cluster = TextCluster(
            centroid_index=0,
            member_indexes=list(range(len(texts))),
            vectorizer=vectorizer,
            cluster_vector=np.asarray(X.mean(axis=0)).ravel(),
        )
        cluster.terms = _top_terms_for_vector(cluster.cluster_vector, vectorizer, top_n=5)
        return [cluster]

    km = KMeans(n_clusters=chosen_k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)

    out: list[TextCluster] = []
    for cid in range(chosen_k):
        members = [i for i, lab in enumerate(labels) if lab == cid]
        if not members:
            continue
        centroid = km.cluster_centers_[cid]
        cluster = TextCluster(
            centroid_index=cid,
            member_indexes=members,
            vectorizer=vectorizer,
            cluster_vector=centroid,
        )
        cluster.terms = _top_terms_for_vector(centroid, vectorizer, top_n=5)
        out.append(cluster)
    return out


def _top_terms_for_vector(
    vec: np.ndarray, vectorizer: TfidfVectorizer, *, top_n: int
) -> list[str]:
    vec = np.asarray(vec).ravel()
    if vec.size == 0:
        return []
    top_idx = np.argsort(-vec)[: top_n * 2]
    feature_names = vectorizer.get_feature_names_out()
    terms: list[str] = []
    for i in top_idx:
        if i < len(feature_names) and vec[i] > 0:
            terms.append(str(feature_names[i]))
        if len(terms) >= top_n:
            break
    return terms


def summarize_cluster_terms(cluster: TextCluster, *, top_n: int = 5) -> list[str]:
    if cluster.terms:
        return cluster.terms[:top_n]
    if cluster.vectorizer is not None and cluster.cluster_vector is not None:
        return _top_terms_for_vector(cluster.cluster_vector, cluster.vectorizer, top_n=top_n)
    return []

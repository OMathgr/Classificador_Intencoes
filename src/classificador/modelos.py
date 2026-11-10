"""Pipelines de classificação: TF-IDF + modelos lineares do scikit-learn."""

from __future__ import annotations

import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def normalizar(texto: str) -> str:
    """Minúsculas e sem acentos, para "pôr" e "por" virarem o mesmo termo."""
    sem_acento = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in sem_acento if not unicodedata.combining(c))


def _vetorizador() -> FeatureUnion:
    """Junta n-gramas de palavras (1 a 2) e de caracteres (2 a 5).

    Os n-gramas de caracteres ajudam com erros de digitação e variações de
    palavras ("lembra", "lembrete", "lembrar"), comuns em mensagens curtas.
    """
    palavras = TfidfVectorizer(
        preprocessor=normalizar, analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True
    )
    caracteres = TfidfVectorizer(
        preprocessor=normalizar, analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True
    )
    return FeatureUnion([("palavras", palavras), ("caracteres", caracteres)])


MODELOS = {
    "naive_bayes": lambda: ComplementNB(alpha=0.3),
    "regressao_logistica": lambda: LogisticRegression(C=10.0, max_iter=2000),
    "svm_linear": lambda: LinearSVC(C=0.5),
}


def construir(nome: str) -> Pipeline:
    """Cria o pipeline completo (vetorização + classificador) pelo nome."""
    if nome not in MODELOS:
        raise ValueError(f"Modelo desconhecido: {nome}. Opções: {sorted(MODELOS)}")
    return Pipeline([("tfidf", _vetorizador()), ("clf", MODELOS[nome]())])

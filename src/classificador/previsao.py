"""Previsão de intenção com as k opções mais prováveis."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

CAMINHO_MODELO = Path(__file__).resolve().parents[2] / "models" / "modelo.joblib"


@dataclass(frozen=True)
class Palpite:
    intencao: str
    pontuacao: float


def _softmax(valores: np.ndarray) -> np.ndarray:
    exp = np.exp(valores - valores.max())
    return exp / exp.sum()


def top_k(modelo: Pipeline, texto: str, k: int = 3) -> list[Palpite]:
    """Devolve as k intenções mais prováveis, da maior para a menor.

    Modelos com probabilidade (regressão logística, Naive Bayes) usam
    predict_proba. O SVM linear não calcula probabilidade, então a pontuação
    é um softmax das distâncias ao hiperplano: serve para ordenar, não como
    probabilidade calibrada.
    """
    if not texto or not texto.strip():
        raise ValueError("Texto vazio: não há o que classificar.")
    if hasattr(modelo, "predict_proba"):
        pontuacoes = modelo.predict_proba([texto])[0]
    else:
        pontuacoes = _softmax(modelo.decision_function([texto])[0])
    ordem = np.argsort(pontuacoes)[::-1][:k]
    classes = modelo.classes_
    return [Palpite(str(classes[i]), float(pontuacoes[i])) for i in ordem]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Classifica a intenção de uma mensagem.")
    parser.add_argument("texto", help="Mensagem a classificar, entre aspas.")
    parser.add_argument("-k", type=int, default=3, help="Quantas intenções mostrar.")
    parser.add_argument("--modelo", default=str(CAMINHO_MODELO), help="Arquivo .joblib do modelo.")
    args = parser.parse_args(argv)

    caminho = Path(args.modelo)
    if not caminho.exists():
        raise SystemExit(f"Modelo não encontrado em {caminho}. Rode antes: python -m classificador.treino")
    modelo = joblib.load(caminho)
    for palpite in top_k(modelo, args.texto, args.k):
        print(f"{palpite.intencao:<28} {palpite.pontuacao:.3f}")


if __name__ == "__main__":
    main()

"""Treina e compara modelos, escolhe o melhor na validação e avalia no teste."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from classificador import dados, modelos

RAIZ = Path(__file__).resolve().parents[2]


def avaliar(y_real, y_prev) -> dict[str, float]:
    """Acurácia e F1 macro. O F1 macro dá o mesmo peso a intenções raras e comuns."""
    return {
        "acuracia": round(accuracy_score(y_real, y_prev), 4),
        "f1_macro": round(f1_score(y_real, y_prev, average="macro", zero_division=0), 4),
    }


def principais_confusoes(y_real, y_prev, rotulos, n: int = 10) -> list[dict]:
    """Pares (real -> previsto) que o modelo mais confunde no teste."""
    matriz = confusion_matrix(y_real, y_prev, labels=rotulos)
    pares = Counter()
    for i, real in enumerate(rotulos):
        for j, previsto in enumerate(rotulos):
            if i != j and matriz[i, j]:
                pares[(real, previsto)] = int(matriz[i, j])
    return [{"real": r, "previsto": p, "quantidade": q} for (r, p), q in pares.most_common(n)]


def treinar(caminho_dados: Path, pasta_modelos: Path, pasta_relatorios: Path) -> dict:
    df = dados.carregar(caminho_dados)
    treino, validacao, teste = dados.dividir(df)

    resultados = {}
    baseline = DummyClassifier(strategy="most_frequent").fit(treino["texto"], treino["intencao"])
    resultados["baseline_mais_frequente"] = avaliar(validacao["intencao"], baseline.predict(validacao["texto"]))

    treinados = {}
    for nome in modelos.MODELOS:
        inicio = time.perf_counter()
        pipeline = modelos.construir(nome).fit(treino["texto"], treino["intencao"])
        metricas = avaliar(validacao["intencao"], pipeline.predict(validacao["texto"]))
        metricas["segundos_treino"] = round(time.perf_counter() - inicio, 1)
        resultados[nome] = metricas
        treinados[nome] = pipeline
        print(f"{nome:<22} validação: acurácia={metricas['acuracia']:.3f}  F1 macro={metricas['f1_macro']:.3f}")

    melhor = max(treinados, key=lambda n: resultados[n]["f1_macro"])
    print(f"\nMelhor na validação: {melhor}")

    # Modelo final: treina de novo com treino + validação e mede só uma vez no teste.
    final = modelos.construir(melhor).fit(
        list(treino["texto"]) + list(validacao["texto"]),
        list(treino["intencao"]) + list(validacao["intencao"]),
    )
    previsto = final.predict(teste["texto"])
    metricas_teste = avaliar(teste["intencao"], previsto)
    print(f"{melhor:<22} TESTE:     acurácia={metricas_teste['acuracia']:.3f}  F1 macro={metricas_teste['f1_macro']:.3f}")

    rotulos = sorted(df["intencao"].unique())
    relatorio = {
        "base": "MASSIVE 1.1, pt-PT",
        "tamanhos": {"treino": len(treino), "validacao": len(validacao), "teste": len(teste)},
        "num_intencoes": len(rotulos),
        "validacao": resultados,
        "modelo_escolhido": melhor,
        "teste": metricas_teste,
        "principais_confusoes_teste": principais_confusoes(teste["intencao"], previsto, rotulos),
    }

    pasta_modelos.mkdir(parents=True, exist_ok=True)
    pasta_relatorios.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, pasta_modelos / "modelo.joblib")
    (pasta_relatorios / "metricas.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    (pasta_relatorios / "relatorio_por_intencao.txt").write_text(
        classification_report(teste["intencao"], previsto, zero_division=0), encoding="utf-8"
    )
    return relatorio


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Treina o classificador de intenções.")
    parser.add_argument("--dados", default=str(dados.CAMINHO_PADRAO))
    parser.add_argument("--modelos", default=str(RAIZ / "models"))
    parser.add_argument("--relatorios", default=str(RAIZ / "reports"))
    args = parser.parse_args(argv)
    treinar(Path(args.dados), Path(args.modelos), Path(args.relatorios))


if __name__ == "__main__":
    main()

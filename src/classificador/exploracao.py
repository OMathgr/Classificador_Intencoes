"""Análise exploratória (EDA) da base, salva em reports/exploracao.md."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from classificador import dados

RAIZ = Path(__file__).resolve().parents[2]


def resumo(df: pd.DataFrame) -> str:
    treino = df[df["particao"] == "train"]
    contagem = treino["intencao"].value_counts()
    palavras = df["texto"].str.split().str.len()
    duplicadas = df.duplicated(subset=["texto"]).sum()
    conflitos = df.groupby("texto")["intencao"].nunique()
    conflitos = int((conflitos > 1).sum())

    linhas = [
        "# Análise exploratória: MASSIVE 1.1 (pt-PT)",
        "",
        f"- Frases: {len(df)} ({', '.join(f'{p}: {n}' for p, n in df['particao'].value_counts().items())})",
        f"- Intenções: {df['intencao'].nunique()} em {df['cenario'].nunique()} cenários",
        f"- Palavras por frase: média {palavras.mean():.1f}, mediana {palavras.median():.0f}, máximo {palavras.max()}",
        f"- Frases repetidas (mesmo texto): {duplicadas}",
        f"- Textos idênticos com intenções diferentes: {conflitos}",
        "",
        "## Desbalanceamento no treino",
        "",
        f"A intenção mais comum tem {contagem.iloc[0]} exemplos e a mais rara tem {contagem.iloc[-1]}"
        f" (razão de {contagem.iloc[0] / contagem.iloc[-1]:.0f}x). Por isso a métrica principal é o F1 macro,"
        " que dá o mesmo peso a todas as intenções.",
        "",
        "| Mais comuns | Exemplos | Mais raras | Exemplos |",
        "|---|---|---|---|",
    ]
    comuns = contagem.head(5)
    raras = contagem.tail(5)[::-1]
    for (c, nc), (r, nr) in zip(comuns.items(), raras.items()):
        linhas.append(f"| {c} | {nc} | {r} | {nr} |")

    linhas += ["", "## Exemplos por cenário", ""]
    for cenario, grupo in df.groupby("cenario"):
        exemplo = grupo.iloc[0]
        n = grupo["intencao"].nunique()
        linhas.append(f"- **{cenario}** ({n} {'intenção' if n == 1 else 'intenções'}):\"{exemplo['texto']}\" -> `{exemplo['intencao']}`")
    return "\n".join(linhas) + "\n"


def main() -> None:
    df = dados.carregar()
    texto = resumo(df)
    destino = RAIZ / "reports" / "exploracao.md"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()

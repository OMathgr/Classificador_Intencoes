"""Carregamento e validação da base MASSIVE (pt-PT)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUNAS_OBRIGATORIAS = ("utt", "intent", "partition", "scenario")
PARTICOES_VALIDAS = {"train", "dev", "test"}
CAMINHO_PADRAO = Path(__file__).resolve().parents[2] / "data" / "raw" / "pt-PT.jsonl"


class DadosInvalidosError(ValueError):
    """Erro levantado quando a base não tem o formato esperado."""


def validar(df: pd.DataFrame) -> pd.DataFrame:
    """Confere colunas, partições e textos vazios; devolve uma cópia limpa.

    Linhas com texto vazio ou intenção ausente são descartadas, porque não servem
    para treino nem avaliação. Colunas faltando ou partições desconhecidas são
    erro, porque indicam um arquivo diferente do esperado.
    """
    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        raise DadosInvalidosError(f"Colunas obrigatórias ausentes: {faltando}")

    desconhecidas = set(df["partition"].dropna().unique()) - PARTICOES_VALIDAS
    if desconhecidas:
        raise DadosInvalidosError(f"Partições desconhecidas: {sorted(desconhecidas)}")

    limpo = df.copy()
    limpo["utt"] = limpo["utt"].astype("string").str.strip()
    mascara_valida = limpo["utt"].notna() & (limpo["utt"] != "") & limpo["intent"].notna()
    return limpo.loc[mascara_valida].reset_index(drop=True)


def carregar(caminho: str | Path = CAMINHO_PADRAO) -> pd.DataFrame:
    """Lê o arquivo JSONL do MASSIVE e devolve só as colunas usadas no projeto."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Base não encontrada em {caminho}. Rode: python scripts/baixar_dados.py"
        )
    df = pd.read_json(caminho, lines=True)
    df = validar(df)
    return df[list(COLUNAS_OBRIGATORIAS)].rename(
        columns={"utt": "texto", "intent": "intencao", "partition": "particao", "scenario": "cenario"}
    )


def dividir(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separa treino, validação (dev) e teste usando a divisão oficial da base."""
    return (
        df[df["particao"] == "train"].reset_index(drop=True),
        df[df["particao"] == "dev"].reset_index(drop=True),
        df[df["particao"] == "test"].reset_index(drop=True),
    )

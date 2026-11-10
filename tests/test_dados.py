import pandas as pd
import pytest

from classificador import dados


def test_validar_descarta_texto_vazio_e_intencao_ausente(base_pequena):
    sujo = pd.concat([
        base_pequena,
        pd.DataFrame([
            {"utt": "   ", "intent": "alarm_set", "partition": "train", "scenario": "alarm"},
            {"utt": "olá", "intent": None, "partition": "train", "scenario": "general"},
        ]),
    ])
    limpo = dados.validar(sujo)
    assert len(limpo) == len(base_pequena)


def test_validar_tira_espacos_das_pontas():
    df = pd.DataFrame([{"utt": "  vai chover  ", "intent": "weather_query", "partition": "train", "scenario": "weather"}])
    assert dados.validar(df)["utt"].iloc[0] == "vai chover"


def test_validar_recusa_coluna_faltando(base_pequena):
    with pytest.raises(dados.DadosInvalidosError, match="intent"):
        dados.validar(base_pequena.drop(columns=["intent"]))


def test_validar_recusa_particao_desconhecida(base_pequena):
    base_pequena.loc[0, "partition"] = "validacao"
    with pytest.raises(dados.DadosInvalidosError, match="validacao"):
        dados.validar(base_pequena)


def test_carregar_renomeia_colunas_e_dividir_separa_particoes(tmp_path, base_pequena):
    arquivo = tmp_path / "pt-PT.jsonl"
    base_pequena.to_json(arquivo, orient="records", lines=True, force_ascii=False)
    df = dados.carregar(arquivo)
    assert list(df.columns) == ["texto", "intencao", "particao", "cenario"]
    treino, validacao, teste = dados.dividir(df)
    assert (len(treino), len(validacao), len(teste)) == (9, 0, 3)


def test_carregar_arquivo_inexistente_explica_como_baixar(tmp_path):
    with pytest.raises(FileNotFoundError, match="baixar_dados"):
        dados.carregar(tmp_path / "nao_existe.jsonl")

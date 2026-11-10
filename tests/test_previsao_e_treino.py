import json

import joblib
import pytest

from classificador import modelos, previsao, treino


@pytest.fixture
def modelos_treinados(base_pequena):
    dados_treino = base_pequena[base_pequena["partition"] == "train"]
    return {
        nome: modelos.construir(nome).fit(dados_treino["utt"], dados_treino["intent"])
        for nome in ("regressao_logistica", "svm_linear")
    }


@pytest.mark.parametrize("nome", ["regressao_logistica", "svm_linear"])
def test_top_k_ordena_e_soma_no_maximo_um(nome, modelos_treinados):
    palpites = previsao.top_k(modelos_treinados[nome], "toca uma música de rock", k=3)
    assert len(palpites) == 3
    assert palpites[0].intencao == "play_music"
    pontuacoes = [p.pontuacao for p in palpites]
    assert pontuacoes == sorted(pontuacoes, reverse=True)
    assert sum(pontuacoes) <= 1.0 + 1e-9


def test_top_k_recusa_texto_vazio(modelos_treinados):
    with pytest.raises(ValueError, match="vazio"):
        previsao.top_k(modelos_treinados["regressao_logistica"], "   ")


def test_cli_de_previsao_imprime_intencoes(tmp_path, modelos_treinados, capsys):
    caminho = tmp_path / "modelo.joblib"
    joblib.dump(modelos_treinados["regressao_logistica"], caminho)
    previsao.main(["vai chover amanhã", "-k", "2", "--modelo", str(caminho)])
    saida = capsys.readouterr().out.strip().splitlines()
    assert len(saida) == 2
    assert saida[0].startswith("weather_query")


def test_principais_confusoes_conta_so_erros():
    reais = ["a", "a", "b", "b", "b"]
    previstos = ["a", "b", "a", "a", "b"]
    confusoes = treino.principais_confusoes(reais, previstos, ["a", "b"])
    assert confusoes[0] == {"real": "b", "previsto": "a", "quantidade": 2}
    assert {"real": "a", "previsto": "b", "quantidade": 1} in confusoes


def test_treino_completo_gera_modelo_e_relatorios(tmp_path, base_pequena):
    # Base pequena com as três partições, para rodar o fluxo inteiro rápido.
    base = base_pequena.copy()
    base.loc[base.index % 4 == 2, "partition"] = "dev"
    arquivo = tmp_path / "pt-PT.jsonl"
    base.to_json(arquivo, orient="records", lines=True, force_ascii=False)

    relatorio = treino.treinar(arquivo, tmp_path / "models", tmp_path / "reports")

    assert relatorio["modelo_escolhido"] in modelos.MODELOS
    assert (tmp_path / "models" / "modelo.joblib").exists()
    salvo = json.loads((tmp_path / "reports" / "metricas.json").read_text(encoding="utf-8"))
    assert salvo["num_intencoes"] == 3
    assert "baseline_mais_frequente" in salvo["validacao"]

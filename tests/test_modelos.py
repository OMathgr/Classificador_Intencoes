import pytest

from classificador import modelos


def test_normalizar_remove_acentos_e_maiusculas():
    assert modelos.normalizar("Põe a Música") == "poe a musica"


@pytest.mark.parametrize("nome", sorted(modelos.MODELOS))
def test_cada_modelo_aprende_intencoes_simples(nome, base_pequena):
    treino = base_pequena[base_pequena["partition"] == "train"]
    pipeline = modelos.construir(nome).fit(treino["utt"], treino["intent"])
    assert pipeline.predict(["acorda-me às nove"])[0] == "alarm_set"
    assert pipeline.predict(["vai estar sol amanhã"])[0] == "weather_query"


def test_modelo_desconhecido_lista_opcoes():
    with pytest.raises(ValueError, match="svm_linear"):
        modelos.construir("random_forest")

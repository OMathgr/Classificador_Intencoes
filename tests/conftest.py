import pandas as pd
import pytest


@pytest.fixture
def base_pequena() -> pd.DataFrame:
    """Base mínima no formato do MASSIVE, com 3 intenções bem separadas."""
    exemplos = {
        "alarm_set": ["acorda-me às sete", "põe um alarme para as oito", "alarme às seis da manhã", "acorda me amanhã cedo"],
        "weather_query": ["vai chover hoje", "como está o tempo", "previsão do tempo para amanhã", "está calor lá fora"],
        "play_music": ["toca uma música", "quero ouvir rock", "põe a tocar samba", "toca a minha playlist"],
    }
    linhas = []
    for intencao, frases in exemplos.items():
        for i, frase in enumerate(frases):
            linhas.append({
                "utt": frase,
                "intent": intencao,
                "partition": "test" if i == 3 else "train",
                "scenario": intencao.split("_")[0],
            })
    return pd.DataFrame(linhas)

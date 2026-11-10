# Classificador de Intenções

Classificador de intenções de mensagens em português com **scikit-learn**. Dada uma frase como *"quero cancelar o alarme de amanhã"*, o modelo diz o que a pessoa quer (`alarm_remove`), com as três intenções mais prováveis e uma pontuação para cada. É o mesmo problema que um agente conversacional resolve antes de responder.

```
$ python -m classificador.previsao "quero cancelar o alarme de amanhã"
alarm_remove                 0.956
calendar_remove              0.019
alarm_query                  0.011
```

## Resultados

Base: **MASSIVE 1.1 (Amazon), português de Portugal (pt-PT)**. São 16.521 frases e 60 intenções em 18 cenários (alarme, música, clima, agenda, e-mail, transporte etc.), já divididas pela própria base em treino (11.514), validação (2.033) e teste (2.974).

Os três modelos foram comparados na **validação**. O melhor foi retreinado com treino + validação e medido **uma única vez no teste**:

| Modelo | Acurácia (validação) | F1 macro (validação) | Tempo de treino |
|---|---|---|---|
| Baseline (sempre a intenção mais comum) | 6,4% | 0,002 | - |
| Naive Bayes (ComplementNB) | 77,7% | 0,707 | 2 s |
| SVM linear (LinearSVC) | 83,9% | 0,807 | 5 s |
| **Regressão logística** | **83,7%** | **0,814** | 40 s |

**No teste, a regressão logística teve 85,5% de acurácia e F1 macro de 0,818.**

O critério de escolha foi o **F1 macro**, não a acurácia, porque a base é desbalanceada: a intenção mais comum tem 810 exemplos no treino e a mais rara tem 4. O F1 macro dá o mesmo peso a todas as intenções, então um modelo que só acerta as comuns não sai bem nessa métrica. O SVM teve acurácia um pouco maior, mas F1 macro menor.

### Onde o modelo erra

As confusões mais frequentes no teste fazem sentido:

- `general_quirky` (conversa solta) ↔ `qa_factoid` (perguntas de conhecimento geral): 31 erros nos dois sentidos. As duas classes têm frases abertas e parecidas.
- `calendar_set` ↔ `calendar_query`: 23 erros. "Marcar" e "consultar" um compromisso usam quase as mesmas palavras.
- Intenções com pouquíssimos exemplos no teste (`general_greet`, `iot_hue_lighton`, `music_dislikeness`) têm F1 baixo e números instáveis.

Relatório completo por intenção: [`reports/relatorio_por_intencao.txt`](reports/relatorio_por_intencao.txt). Métricas e confusões: [`reports/metricas.json`](reports/metricas.json). Análise exploratória: [`reports/exploracao.md`](reports/exploracao.md).

## Como funciona

1. **Dados** (`src/classificador/dados.py`): lê o JSONL, confere colunas e partições, tira espaços das pontas e descarta frases vazias ou sem intenção.
2. **Atributos** (`src/classificador/modelos.py`): TF-IDF com n-gramas de **palavras** (1 a 2) e de **caracteres** (2 a 5), sobre o texto em minúsculas e sem acentos. Os n-gramas de caracteres ajudam com erros de digitação e variações da mesma palavra ("acorda", "acordar", "acorda-me").
3. **Modelos**: Naive Bayes, regressão logística e SVM linear, cada um num `Pipeline` do scikit-learn junto com a vetorização. Assim o mesmo objeto recebe texto bruto no treino e na previsão.
4. **Treino e avaliação** (`src/classificador/treino.py`): compara na validação, escolhe pelo F1 macro, retreina e avalia no teste. Salva o modelo em `models/modelo.joblib`.
5. **Previsão** (`src/classificador/previsao.py`): devolve as k intenções mais prováveis. No SVM, que não calcula probabilidade, a pontuação é um softmax das distâncias e serve só para ordenar.

## Como rodar

Requer Python 3.10 ou mais novo.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (no Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"

python scripts/baixar_dados.py  # baixa o MASSIVE (cerca de 40 MB) e guarda só o pt-PT
python -m classificador.exploracao
python -m classificador.treino
python -m classificador.previsao "põe uma música do caetano"
pytest
```

## Testes

17 testes com **pytest** cobrem a validação dos dados, a normalização do texto, o aprendizado de cada modelo numa base mínima, a ordenação do top-k, o comando de previsão e o fluxo de treino completo, que gera o modelo e os relatórios.

## Limitações e próximos passos

- A base é **português de Portugal** ("acorda-me", "pôr a tocar"). Frases em português do Brasil funcionam na maior parte dos casos, mas o vocabulário não é o mesmo.
- É um modelo de "saco de palavras": não entende ordem nem contexto. O próximo passo natural é comparar com embeddings de um modelo de linguagem multilíngue e com um classificador ajustado (fine-tuning).
- Adicionar uma classe "fora do escopo", usando um limiar de confiança, para o modelo responder "não entendi" em vez de sempre escolher uma intenção.

## Dados e licença

A base **MASSIVE** é da Amazon, licenciada sob [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Os textos de licença e aviso estão em `data/raw/` depois do download.

> FitzGerald et al. (2022). *MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages.*

Os dados não ficam no repositório. O script `scripts/baixar_dados.py` baixa a base do servidor oficial.

Desenvolvido com o **Claude Code** como par de programação.

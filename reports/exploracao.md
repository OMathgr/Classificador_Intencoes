# Análise exploratória: MASSIVE 1.1 (pt-PT)

- Frases: 16521 (train: 11514, test: 2974, dev: 2033)
- Intenções: 60 em 18 cenários
- Palavras por frase: média 7.1, mediana 7, máximo 70
- Frases repetidas (mesmo texto): 802
- Textos idênticos com intenções diferentes: 31

## Desbalanceamento no treino

A intenção mais comum tem 810 exemplos e a mais rara tem 4 (razão de 202x). Por isso a métrica principal é o F1 macro, que dá o mesmo peso a todas as intenções.

| Mais comuns | Exemplos | Mais raras | Exemplos |
|---|---|---|---|
| calendar_set | 810 | cooking_query | 4 |
| play_music | 639 | music_dislikeness | 14 |
| weather_query | 573 | audio_volume_other | 18 |
| calendar_query | 566 | iot_hue_lighton | 22 |
| general_quirky | 555 | general_greet | 25 |

## Exemplos por cenário

- **alarm** (3 intenções):"acorda-me às cinco da manhã durante esta semana" -> `alarm_set`
- **audio** (4 intenções):"silêncio" -> `audio_volume_mute`
- **calendar** (3 intenções):"verifique quando o show inicia" -> `calendar_query`
- **cooking** (2 intenções):"pode sugerir-me um jantar de comida saudável" -> `cooking_recipe`
- **datetime** (2 intenções):"que horas são na austrália" -> `datetime_query`
- **email** (4 intenções):"não queria que enviasse esse texto mas que esperasse até que eu dissesse para enviar" -> `email_sendemail`
- **general** (3 intenções):"verifica se o meu carro está pronto" -> `general_quirky`
- **iot** (9 intenções):"cor de rosa é tudo que precisamos" -> `iot_hue_lightchange`
- **lists** (3 intenções):"acabei a minha lista de tarefas" -> `lists_query`
- **music** (4 intenções):"gosto de música rock" -> `music_likeness`
- **news** (1 intenção):"o que está a acontecer no futebol hoje" -> `news_query`
- **play** (5 intenções):"quero ouvir eu tenho dois amores do marco paulo novamente" -> `play_music`
- **qa** (5 intenções):"mantém me atualizado dos preços do mercado bolsista" -> `qa_stock`
- **recommendation** (3 intenções):"encontra me um bom restaurante para jantar" -> `recommendation_locations`
- **social** (2 intenções):"leva-me ao site do huffington post" -> `social_post`
- **takeaway** (2 intenções):"olly mostra uma lista das opções de entrega de comida chinesa com mais avaliações" -> `takeaway_query`
- **transport** (4 intenções):"você precisa me dar direções diferentes" -> `transport_query`
- **weather** (1 intenção):"está a chover" -> `weather_query`

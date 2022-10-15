# COVID-19 em Minas Gerais

![](https://github.com/fernandarrios/tcc_covid_recuperacao/blob/main/covid_mg_banner.png)

Segundo a Organização Pan-Americana de Saúde (OPAS), em 31 de dezembro de 2019, a Organização Mundial de Saúde (OMS) foi alertada sobre diversos casos de pneumonia na cidade de Wuhan, província de Hubei, na República Popular da China. Os casos investigados foram causados por uma nova cepa viral ainda não identificada entre os seres humanos, classificada como coronavírus humano SARS-CoV-2. 

Ao todo são sete coronavírus humanos (HCoVs) identificados e classificados: HCoV-229E, HCoV-OC43, HCoV-NL63, HCoV-HKU1, SARS-COV, MERS-COV e o mais recente, o SARS-CoV-2, tido como o novo coronavírus e que causa uma síndrome respiratória aguda grave e que foi responsável por causar a COVID-19.

Em 30 de janeiro de 2020, a OMS declarou que o surto do novo coronavírus constitui uma Emergência de Saúde Pública de Importância Internacional (ESPII) – o mais alto nível de alerta da Organização, conforme previso no Regulamento Sanitário Internacional. Em 11 de março de 2020, a COVID-19 (SARS-CoV-2) foi caracterizada pela OMS como uma pandemia.
Até 21 de setembro de 2022, houve aproximadamente 613 milhões de casos, com 6,53 milhões de mortes no mundo inteiro. No Brasil foram 34,6 milhões de casos, com 685,5 mil mortes. Em Minas Gerais, houve 3,9 milhões de casos com 63,8 mil mortes.

Esse trabalho busca analisar os dados sobre a COVID-19 no Estado de Minas Gerais, desde 01 de janeiro 2020 até 17 de setembro de 2022, disponíveis nos sites do Governo Federal e Governo de Minas Gerais, buscando entender como a pandemia afetou a vida dos mineiros e as características dessas pessoas e construir um modelo que consiga prever quais as chances de alguém que contraiu COVID-19 tem de se recuperar, de morrer ou ficar em recuperação.  por meio modelos de Machine Learning de Classificação, usando como variável alvo a informação “Evolução” do paciente, presente no dataset, sendo 1 para “Recuperado”, 2 para “Em acompanhamento” e 3 para “Óbito”.

### Índice:
1. COVID-19.
   - Objetivo do trabalho.
2. Coleta de dados.
3. Importação dos dados.
4. Tratamento dos dados.
5. Análise Exploratória.
   - Macrorregião.
   - Unidade Regional de Saúde.
   - Portal de Saúde.
   - Casos por decorrer do tempo.
   - Faixa Etária.
   - Comorbidade.
   - Raça.
   - Casos de Internação e Casos de UTI.
   - Evolução de saúde dos casos.
 6. Preparação do modelo.
 7. Treinamento do modelo.
   - Decision Tree Classifier.
   - Random Forest Classifier.
   - Light Gradient Boosting Machine.
 8. Comparação entre os modelos.

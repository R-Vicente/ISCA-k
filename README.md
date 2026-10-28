# ISCA-k: Instance-based Similarity and Contextual Adaptation

Este repositório reúne o necessário para repetir os resultados da dissertação de mestrado em Análise de Dados e Sistemas de Apoio à Decisão (Coimbra Business School | ISCAC, 2026), *ISCA-k: um método adaptativo de imputação por vizinhos mais próximos para dados mistos*. O ISCA-k (Instance-based Similarity and Contextual Adaptation, com k vizinhos) é um método de imputação por vizinhos mais próximos para dados mistos, em que a semelhança entre instâncias se ajusta ao contexto de cada célula a imputar.

O repositório inclui:

- o método;
- os cinco comparadores;
- os 12 datasets congelados, as máscaras de ausência e os resultados de todas as execuções;
- os scripts que geram as tabelas e figuras do capítulo de resultados.

Autor: Ricardo Vicente. Orientação: Francisco Antunes.

[![DOI](https://zenodo.org/badge/1434701897.svg)]
(https://doi.org/10.5281/zenodo.23672166)

## Estrutura

```
README.md
requirements.txt                versões exatas usadas em todas as execuções
reproduzir.py                   estudo principal: uma imputação com qualquer método, dataset, condição e repetição
reproduzir_complementares.py    ablação, sensibilidade a k e recalibração do ISCA-k
reproduzir_distribuicoes.py     preservação das distribuições (MCAR 20% e 40%)
reproduzir_poda.py              ensaios da poda de candidatos
codigo/
  src/isca_k.py                 o método ISCA-k, na versão avaliada
  src/type_detection.py         deteção e codificação dos tipos de variável, usada pelo método
  src/isca_k_revised_pruning.py versão com a poda revista (secção 5.7.3), usada só no ensaio da poda
  src/missingness_*.py          geradores de ausência em bloco e MAR
  benchmark/bench_utils.py      média/moda, k-NN, MICE, métricas e carregamento dos datasets
  benchmark/bench_harness.py    execução dos seis métodos e cálculo das métricas (NRMSE, PFC)
  benchmark/softimpute.py       SoftImpute com escolha de λ por validação
  benchmark/experiments.py      configuração do ISCA-k, variantes dos estudos complementares e diagnósticos
  benchmark/harness_audit.py    SHA-256 dos snapshots e das máscaras, leitura e escrita dos snapshots
  benchmark/medidas.py          medidas de preservação das distribuições
  benchmark/mask_store.py       geração e verificação das máscaras
  testes/                       testes do SoftImpute
dados/
  manifest.json                 descrição dos 12 datasets, tipos de variável e SHA-256
  tipos_variaveis.json          contrato de tipos (numérica ou categórica) de cada coluna
  snapshots/                    tabelas completas, em .npz
  mascaras/                     75 máscaras por dataset e condição (8 condições), com manifesto
resultados/                     resultados de todas as execuções (ver abaixo)
analise/
  analise_tese.py               tabelas e testes do capítulo 5, a partir de resultados/
  figuras.py                    Figuras 5.1 a 5.3
  saida/                        o que estes dois scripts produzem
```

## Ambiente

Todas as execuções da dissertação correram em Python 3.13.3, num computador com Windows e processador Intel com AVX2. As versões dos pacotes estão em `requirements.txt`:

```shell
python -m venv .venv
.venv\Scripts\activate          # Windows; em Linux ou macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Os scripts fixam um único thread nas bibliotecas numéricas, como nas execuções originais.

## Desenho do estudo

| | |
|---|---|
| Datasets | Iris, Wine, Diabetes, Breast Cancer, Heart Disease, Abalone, California Housing, Penguins, Titanic, Auto MPG, Ionosphere, Heart Failure |
| Condições | MCAR uniforme a 5, 10, 20, 30, 40 e 50%; ausência em bloco a 20%; MAR a 20% |
| Repetições | 75 por dataset e condição, com sementes 42 a 116 (repetição r usa a semente 42 + r) |
| Métodos | Média/moda, k-NN, MICE, missForest, SoftImpute, ISCA-k |
| Métricas | NRMSE (média, por variável, do RMSE dividido pelo desvio-padrão) e PFC (proporção de categorias erradas) |
| Total | 12 × 8 × 6 × 75 = 43 200 imputações |

As máscaras são fornecidas e cada script verifica o SHA-256 do ficheiro e de cada máscara antes de imputar. Os snapshots são as tabelas completas usadas como verdade. Os tipos de variável seguem o protocolo da dissertação: em Auto MPG, a coluna `origin` é categórica.

### Configuração dos métodos

Os pormenores e a fundamentação estão no capítulo 4 da dissertação, Tabela 4.2.

- **Média/moda.** Média nas numéricas e moda nas categóricas.
- **k-NN.** Implementação própria para dados mistos.
  - k é escolhido por validação na grelha {1, 3, 5, 9, 15, 20} até 5 000 linhas.
  - Acima de 5 000 linhas, k é √n arredondado e limitado a [5, 20].
- **MICE.** `IterativeImputer` do scikit-learn, com o máximo de iterações escolhido por validação em {5, 10, 20}.
- **missForest.** Implementação própria em Python (`_impute_missforest_audited` em `bench_harness.py`).
  - Florestas do scikit-learn com 100 árvores, até 10 iterações.
  - Paragem quando a variação relativa das imputações numéricas fica abaixo de 10⁻⁴.
  - Afasta-se do pacote R dos autores na regra de paragem e no número de variáveis por divisão nas árvores de regressão.
- **SoftImpute.** `softimpute.py`, com λ = c·λ₀ e c escolhido por validação em {1/1,5; 1/2; 1/5; 1/10; 1/20; 1/50; 1/100}.
- **ISCA-k.** Configuração de referência, definida em `experiments.fit_iscak`:

```python
from isca_k import ISCAkConfig, ISCAkImputer
config = ISCAkConfig(adaptive_k=True, k_min=3, k_max=25, use_mi_weights=True, mi_n_neighbors=5,
                     mi_random_state=42, use_propagation=True, use_distance_shrinkage=True,
                     shrink_strength=0.5, standardize_numeric=True, homogeneity_threshold=2.5,
                     stability_threshold=0.01, use_candidate_pruning=True, candidate_k=25,
                     force_categorical=None, interactive_type_detection=False, verbose=False)
df_imputado = ISCAkImputer(config).fit_transform(df)   # df: pandas.DataFrame com NaN nos valores em falta
```

`force_categorical` deve indicar as colunas categóricas codificadas como números, por exemplo `["origin"]` em Auto MPG.

## Como reproduzir

Estudo principal (uma linha por imputação, com NRMSE, PFC e tempo):

```shell
python reproduzir.py --dataset Iris --condicao mcar_20 --metodo todos --repeticao 0
python reproduzir.py --dataset "Heart Disease" --condicao blockwise_20 --metodo missForest --repeticao 0-74 --csv hd.csv
```

Estudos complementares, distribuições e poda:

```shell
python reproduzir_complementares.py --estudo ablacao --variante sem_shrinkage --dataset Iris --repeticao 0-4
python reproduzir_complementares.py --estudo sensibilidade_k --variante k_25 --dataset "California Housing" --condicao mcar_40
python reproduzir_distribuicoes.py --dataset Wine --condicao mcar_40 --metodo todos --repeticao 0
python reproduzir_poda.py --dataset "California Housing" --variante todas --repeticao 0-4
```

Tabelas e figuras do capítulo 5, a partir dos resultados publicados:

```shell
python analise/analise_tese.py
python analise/figuras.py
```

Cada linha produzida pelos scripts `reproduzir*` pode ser comparada com a linha correspondente em `resultados/` (mesmo dataset, condição, método ou variante e repetição).

## Reprodutibilidade numérica

Todos os métodos são deterministas. Mesmo assim, há diferenças conforme o processador.

**Média/moda, MICE, missForest e SoftImpute.** Testámos noutro sistema (Linux, processador com AVX-512) e reproduzem os resultados publicados, com diferenças da ordem de 10⁻¹⁵.

**k-NN e ISCA-k.** Ordenam distâncias, e quando há distâncias empatadas a ordem dos vizinhos depende da implementação da ordenação. O NumPy 2 escolhe essa implementação conforme as instruções do processador (AVX2 ou AVX-512).

- **Processadores com AVX2, como o das execuções originais.** Obtêm-se os mesmos valores.
- **Processadores com AVX-512.** Desligue essas instruções no NumPy antes de correr:

  ```shell
  set NPY_DISABLE_CPU_FEATURES=AVX512_SPR AVX512_ICL AVX512_CNL AVX512_CLX AVX512_SKX AVX512F AVX512CD   (Windows)
  export NPY_DISABLE_CPU_FEATURES="AVX512_SPR AVX512_ICL AVX512_CNL AVX512_CLX AVX512_SKX AVX512F AVX512CD"   (Linux/macOS)
  ```

  Com isto, o k-NN reproduziu os valores publicados nos casos testados. No ISCA-k podem ficar pequenas diferenças em datasets com muitos empates, como o Titanic: 33 das 75 repetições de MCAR 20% diferiram, com NRMSE médio de 0,85315 contra 0,85303 (0,014%).

Os tempos dependem do equipamento. As medições da dissertação foram feitas em sequência, nas 5 primeiras máscaras de cada taxa MCAR, com um thread, uma execução de aquecimento não contabilizada e a cache da informação mútua limpa antes de cada medição.

## Resultados

`resultados/estudo_principal/`

- `benchmark_repeticoes.csv`: 43 200 linhas (dataset, condition, method, replicate, nrmse, pfc, actual_rate). `pfc` fica vazio quando o dataset não tem categóricas.
- `benchmark_medias.csv`: média e desvio-padrão das 75 repetições.
- `softimpute_parametros.csv`: fração de λ escolhida, iterações, convergência e posto final de cada execução do SoftImpute.

`resultados/tempos/`: tempo de cada imputação (5 máscaras × 6 taxas MCAR × 12 datasets × 6 métodos) e medianas.

`resultados/estudos_complementares/` (ISCA-k)

- `complementares_repeticoes.csv` e `complementares_medias.csv`.
  - `stage`: `reference` para a configuração completa, `ablation`, `ksens` (sensibilidade a k) e `calibration` (recalibração).
  - `variant`: a variante.
- `efeitos_componentes.csv` e `testes_componentes.csv`: variação percentual do NRMSE por componente e testes de Wilcoxon com Holm.
- `regret.csv`: excesso de NRMSE face ao melhor k da grelha.
- `diagnosticos_*.csv`: k escolhido, motivos de paragem, oráculo numérico e entropia dos pesos.
- `hipoteses_*.csv`: associações exploratórias.
- `datasets.csv`: descritores.

`resultados/distribuicoes/`

- `distribuicoes_repeticoes.csv`: medidas por repetição. `missing_*` refere-se às células ocultadas e `full_*` à coluna completa.
- `distribuicoes_por_variavel_celulas_ocultas.csv`: medidas por variável nas células ocultadas, incluindo `sd_ratio`.
- `dados_observados_repeticoes.csv`: as mesmas medidas para os valores observados, como referência.

`resultados/poda_original/` e `resultados/poda_revista/`: tempos e NRMSE dos ensaios da poda (MCAR 20%, repetições 0 a 4).

`resultados/caracterizacao_datasets/`: descritores por dataset e por variável.

## Como citar

Vicente, R. (2026). *ISCA-k: um método adaptativo de imputação por vizinhos mais próximos para dados mistos* (Dissertação de mestrado em Análise de Dados e Sistemas de Apoio à Decisão, Coimbra Business School | ISCAC). Repositório Científico de Acesso Aberto de Portugal.

## Licença

Código distribuído sob a licença MIT (ver `LICENSE`).

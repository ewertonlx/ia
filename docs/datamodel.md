Show, mestre 🚀
Peguei sua estrutura de tabelas e adaptei o `.md` que você tinha pro formato do **Open Food Facts**, já com os tipos de dados e descrições de cada camada.

Aqui está a versão atualizada:

---

# Data Model: Open Food Facts

Este documento descreve a modelagem de dados em três camadas: **System of Record (SOR)**, **System of Truth (SOT)** e **Specification (SPEC)**.

---

## 1. System of Record (SOR)

**Tabela:** `sor_food`

Representa os dados brutos, exatamente como chegam do arquivo `.csv`. É a primeira camada de armazenamento, garantindo que tenhamos uma cópia fiel dos dados originais.

* **Propósito:** Ingestão e arquivamento dos dados brutos.
* **Estrutura:** Colunas correspondem ao dataset original, sem limpeza ou transformação.

| Coluna                                   | Tipo de Dado (SQL) | Descrição                                                   |
| ---------------------------------------- | ------------------ | ----------------------------------------------------------- |
| code                                     | VARCHAR(255)       | Código único do produto (ID).                               |
| product\_name                            | TEXT               | Nome do produto.                                            |
| nutrition\_score\_fr\_100g               | REAL               | Score nutricional do produto (Nutri-Score, por 100g).       |
| quantity                                 | TEXT               | Quantidade informada na embalagem (ex: "500g", "1L").       |
| fruits\_vegetables\_nuts\_100g           | REAL               | Percentual de frutas, vegetais e nozes por 100g.            |
| fruits\_vegetables\_nuts\_estimate\_100g | REAL               | Estimativa do percentual de frutas/vegetais/nozes por 100g. |
| collagen\_meat\_protein\_ratio\_100g     | REAL               | Razão colágeno/proteína da carne (100g).                    |
| cocoa\_100g                              | REAL               | Percentual de cacau por 100g.                               |
| chlorophyl\_100g                         | REAL               | Quantidade de clorofila por 100g.                           |
| carbon\_footprint\_100g                  | REAL               | Pegada de carbono estimada (100g).                          |
| glycemic\_index\_100g                    | REAL               | Índice glicêmico estimado (100g).                           |
| water\_hardness\_100g                    | REAL               | Dureza da água associada ao produto (100g).                 |

---

## 2. System of Truth (SOT)

**Tabela:** `sot_food`

Esta camada representa a "versão única da verdade". Os dados da SOR são limpos, padronizados e enriquecidos. É a base confiável para análises e modelagem.

* **Propósito:** Fornecer dados limpos e consistentes.
* **Transformações Aplicadas:**

  * Conversão de `quantity` para valor numérico (quando possível).
  * Padronização de colunas nutricionais.
  * Remoção de colunas não essenciais para análise/modelagem.

| Coluna                                   | Tipo de Dado (SQL) | Descrição                                                   |
| ---------------------------------------- | ------------------ | ----------------------------------------------------------- |
| product\_name                            | TEXT               | Nome do produto, padronizado.                               |
| quantity                                 | REAL               | Quantidade em valor numérico (ex: 500).                     |
| fruits\_vegetables\_nuts\_100g           | REAL               | Percentual de frutas/vegetais/nozes por 100g.               |
| fruits\_vegetables\_nuts\_estimate\_100g | REAL               | Estimativa do percentual de frutas/vegetais/nozes por 100g. |
| nutrition\_score\_fr\_100g               | INTEGER            | Nutri-Score (normalizado e convertido para inteiro).        |

---

## 3. Specification (SPEC)

**Tabela:** `spec_food`

Camada final, pronta para ser consumida em modelos de **machine learning**. Contém as variáveis independentes (features) e a variável alvo.

* **Propósito:** Fornecer dataset já limpo e pronto para modelagem.
* **Estrutura:** Geralmente é uma cópia ou visão da `sot_food`.

| Coluna                                   | Tipo de Dado (SQL) | Descrição                                     |
| ---------------------------------------- | ------------------ | --------------------------------------------- |
| product\_name                            | TEXT               | Nome do produto.                              |
| quantity                                 | REAL               | Quantidade numérica.                          |
| fruits\_vegetables\_nuts\_100g           | REAL               | Percentual de frutas/vegetais/nozes por 100g. |
| fruits\_vegetables\_nuts\_estimate\_100g | REAL               | Estimativa de frutas/vegetais/nozes por 100g. |
| nutrition\_score\_fr\_100g               | INTEGER            | Variável alvo (score nutricional).            |

---
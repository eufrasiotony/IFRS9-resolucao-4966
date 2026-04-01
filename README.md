# IFRS 9 — Modelagem de Risco de Crédito com Dados Sintéticos

## O que é IFRS 9?

O **IFRS 9** (International Financial Reporting Standard 9) é o padrão contábil internacional para instrumentos financeiros, adotado no Brasil pelo **CPC 48**. Ele substituiu o IAS 39 e trouxe uma mudança fundamental no cálculo de provisões:

| | IAS 39 (antigo) | IFRS 9 (atual) |
|---|---|---|
| **Modelo** | Perda Incorrida | Perda Esperada (ECL) |
| **Quando reconhecer?** | Após o default | Antes do default |
| **Horizonte** | 12 meses ou mais | Depende do estágio |

---

## Os 3 Estágios do IFRS 9

```
Stage 1  →  Stage 2  →  Stage 3
12m ECL    Lifetime ECL   Lifetime ECL
              (ASRC)      (Crédito Deteriorado / Default)
```

- **Stage 1**: Sem aumento significativo no risco de crédito (ASRC) desde a originação → ECL de 12 meses
- **Stage 2**: Com ASRC mas ainda não em default → ECL vitalício
- **Stage 3**: Crédito deteriorado / em default → ECL vitalício (PD = 100%)

**Critérios de ASRC (migração Stage 1 → 2):**
- Queda > 100 pontos no score de crédito
- Atraso ≥ 30 dias
- DTI elevado combinado com histórico de inadimplência

---

## Os 3 Componentes da ECL

$$\text{ECL} = \text{PD} \times \text{LGD} \times \text{EAD} \times \text{Fator de Desconto}$$

### PD — Probability of Default
> Probabilidade de o devedor não honrar suas obrigações no horizonte definido.

- **PD 12m**: para Stage 1
- **PD Lifetime**: para Stage 2 e 3
- Modelo: **Regressão Logística** + **Gradient Boosting**
- Avaliação: AUC-ROC, Gini, KS statistic

### LGD — Loss Given Default
> Percentual da exposição que **não será recuperado** em caso de default.

$$\text{LGD} = 1 - \text{Taxa de Recuperação}$$

- Depende fortemente do tipo de garantia e do LTV
- Modelo: **Regressão Linear** com transformação logit (variável limitada a [0,1])
- Avaliação: R², MAE, correlação de Spearman

### EAD — Exposure at Default
> Valor da exposição **no momento** do default.

Para créditos rotativos (cartão, limite de crédito):
$$\text{EAD} = \text{Saldo Sacado} + \text{CCF} \times \text{Saldo Não Sacado}$$

- **CCF (Credit Conversion Factor)**: percentual do limite não utilizado que será sacado antes do default
- Modelo: **Regressão Linear** para estimar CCF
- Avaliação: R², MAE

---

## Estrutura do Projeto

```
IFRS9/
├── data/
│   ├── raw/
│   │   └── portfolio.csv          # Carteira sintética gerada
│   └── processed/
│       └── portfolio_final.csv    # Carteira com scores e ECL
├── outputs/
│   ├── models/                    # Modelos salvos (.pkl)
│   ├── plots/                     # Gráficos de validação
│   └── ecl_report.csv             # Relatório de ECL por estágio
├── src/
│   ├── data_generation.py         # Geração de dados sintéticos
│   ├── pd_model.py                # Modelo de PD
│   ├── lgd_model.py               # Modelo de LGD
│   ├── ead_model.py               # Modelo de EAD
│   ├── ecl_calculator.py          # Cálculo da ECL por estágio
│   └── visualization.py           # Gráficos e relatórios visuais
├── main.py                        # Orquestrador do projeto
└── requirements.txt
```

---

## Como Executar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Executar o projeto completo
```bash
python main.py
```

### 3. O que acontece ao executar:
1. Gera carteira sintética com 10.000 contratos
2. Treina modelo de PD (Logistic + GBM) → AUC, Gini, KS
3. Treina modelo de LGD (contratos em default)
4. Treina modelo de EAD / CCF (produtos rotativos)
5. Calcula ECL por estágio (Stage 1, 2 e 3)
6. Salva gráficos em `outputs/plots/`
7. Gera relatório consolidado de ECL

---

## Dados Sintéticos

A carteira simulada contém **10.000 contratos** com:

| Feature | Descrição |
|---|---|
| `credit_score_current` | Score atual do cliente (200-1000) |
| `score_change` | Variação do score desde a originação |
| `debt_to_income` | Razão dívida / renda mensal |
| `ltv_ratio` | Loan-to-Value (saldo / garantia) |
| `num_delinquencies` | Número de inadimplências nos últimos 12m |
| `days_past_due` | Dias em atraso atual |
| `loan_type` | Tipo: Imobiliário, Veículos, Pessoal, Cartão, Empresarial |
| `employment_type` | CLT, CNPJ, Servidor Público, Autônomo, Aposentado |
| `stage` | Estágio IFRS 9 (1, 2 ou 3) |
| `default_flag` | 0 = adimplente, 1 = em default |
| `lgd_true` | LGD real (apenas para validação) |
| `ccf_true` | CCF real (apenas para validação) |

---

## Leitura Recomendada

- [IFRS 9 — IASB](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/)
- [CPC 48 — CFC](https://www.cfc.org.br/)
- [Guia Prático IFRS 9 — EY](https://www.ey.com/ifrs9)
- Basel Committee: "Guidance on credit risk and accounting for expected credit losses"

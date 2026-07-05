# Formulacoes-Agro

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Plataforma de **otimização bayesiana em loop fechado** para formulações agroquímicas. O sistema sugere novas composições experimentais com base em modelos probabilísticos treinados sobre dados históricos de laboratório, permitindo iterar entre simulação, experimento e retreino do modelo.

**Aplicação online:** [https://formulacoes-agro.streamlit.app/](https://formulacoes-agro.streamlit.app/)

---

## Visão geral

O objetivo é encontrar formulações que atinjam propriedades-alvo desejadas — **Viscosidade**, **Escoamento** e **Suspensão (SIM/NÃO)** — a partir das concentrações de 16 insumos. A abordagem combina:

| Componente | Papel |
|------------|--------|
| **GPR** (Gaussian Process Regression) | Prediz Viscosidade e Escoamento com incerteza quantificada |
| **GPC** (Gaussian Process Classifier) | Estima a probabilidade de Suspensão = SIM |
| **Otimização bayesiana** | Recomenda a próxima formulação a ser testada no laboratório |

O fluxo é iterativo: o usuário define os alvos, recebe uma sugestão de composição, executa o experimento, registra os resultados observados e o modelo é retreinado automaticamente.

---

## Funcionalidades

- Carregamento automático do dataset local ou upload manual via interface
- Exportação do dataset atualizado (incluindo novos experimentos)
- Métricas de desempenho: MAE, R² (GPR) e balanced accuracy (GPC), com validação leave-one-out
- Gráficos de diagnóstico: parity plots, efeito de features e matriz de confusão da suspensão
- Função de aquisição customizada que equilibra proximidade ao alvo, exploração e P(Suspensão = SIM)
- Campos de anotação (`Material`, `Característica`) para registro pelo especialista

---

## Estrutura do projeto

```
Formulacoes/
├── app.py           # Interface Streamlit (Module C)
├── model.py         # Pré-processamento, GPR e GPC (Module A)
├── optimizer.py     # Otimização bayesiana (Module B)
├── environment.yml  # Ambiente Conda
├── data/            # Dados locais (não versionados)
│   └── Formulacoes.csv
└── README.md
```

---

## Formato do dataset

O CSV deve conter **21 colunas**, na ordem abaixo:

| Grupo | Colunas |
|-------|---------|
| Anotações | `Material`, `Característica` |
| Features (16) | `Propilenoglicol`, `Glicerina`, `Polietilenoglicol`, `Metilparabeno`, `Sorprophor`, `Geropon DA`, `Antarox`, `Rodasurf`, `Geropon SDS`, `H2O`, `Imidacloprida`, `Amido`, `Goma xantana`, `Alginato`, `Carvão At`, `Biochar` |
| Targets contínuos | `Viscosidade`, `Escoamento` |
| Classificação | `Suspensao` (`SIM` / `NÃO`) |

**Convenções:**

- Features ≥ 0 (zero indica que o insumo não foi utilizado)
- `-1` em Viscosidade ou Escoamento indica valor acima do limite de medição (tratado internamente como valor muito alto)
- Valores ausentes em features numéricas são interpretados como zero

Se o arquivo padrão (`data/Formulacoes.csv`) não estiver disponível, a interface solicita o upload manual.

---

## Instalação

Requisitos: [Conda](https://docs.conda.io/) (ou Miniconda).

```bash
git clone https://github.com/quiles/Formulacoes-Agro.git
cd Formulacoes-Agro

conda env create -f environment.yml
conda activate formulation
```

Coloque o arquivo de formulações em `data/Formulacoes.csv` (ou faça upload pela interface após iniciar o app).

---

## Execução local

```bash
conda activate formulation
streamlit run app.py
```

A aplicação abrirá no navegador (por padrão em `http://localhost:8501`).

---

## Fluxo de uso

1. **Carregar dados** — automaticamente a partir de `data/Formulacoes.csv` ou via upload na barra lateral
2. **Definir alvos** — Viscosidade e Escoamento desejados (padrão: 233,88 e 0,2)
3. **Executar otimização** — obter a composição sugerida e a predição de P(SIM)
4. **Registrar resultado de laboratório** — informar anotações, composição utilizada e valores observados
5. **Retreinar** — o modelo é atualizado e o ciclo pode ser repetido

---

## Dependências principais

- Python 3.11
- scikit-learn (GPR / GPC)
- SciPy (otimização da função de aquisição)
- Streamlit (interface)
- pandas, NumPy, Matplotlib

---

## Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).

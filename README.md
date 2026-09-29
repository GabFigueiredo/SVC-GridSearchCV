# 🧠 Previsão de AVC com SVC + GridSearchCV

Projeto de classificação que usa um **Support Vector Classifier (SVC)** com otimização de hiperparâmetros via **GridSearchCV** para prever se um paciente teve um **AVC (acidente vascular cerebral)**, a partir de dados clínicos e demográficos.

## 📊 Dataset

- **Nome:** [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Fonte:** Kaggle (autor *fedesoriano*)
- **Arquivo:** `healthcare-dataset-stroke-data.csv`
- **Tamanho:** ~5.110 registros e 12 colunas
- **Alvo:** `stroke` (0 = sem AVC, 1 = com AVC)

| Coluna | Tipo | Descrição |
|---|---|---|
| `gender` | Categórica | Gênero do paciente |
| `age` | Numérica | Idade |
| `hypertension` | Binária | Possui hipertensão (0/1) |
| `heart_disease` | Binária | Possui doença cardíaca (0/1) |
| `ever_married` | Categórica | Já foi casado(a) |
| `work_type` | Categórica | Tipo de trabalho |
| `Residence_type` | Categórica | Zona urbana ou rural |
| `avg_glucose_level` | Numérica | Nível médio de glicose |
| `bmi` | Numérica | Índice de massa corporal (contém N/A) |
| `smoking_status` | Categórica | Situação de tabagismo |

## 🛠️ Etapas do projeto

1. **Remoção de duplicidade**: exclusão da coluna `id`, que não tem valor preditivo, e remoção de linhas duplicadas.
2. **Tratamento de valores ausentes**: imputação da mediana em `bmi`. A categoria `Unknown` de `smoking_status` foi mantida como categoria própria.
3. **Tratamento de dados categóricos**: One-Hot Encoding. A categoria rara `Other` (1 registro) foi removida.
4. **Normalização**: `StandardScaler` nas variáveis contínuas, essencial para o SVC, que é baseado em distâncias.
5. **Pipeline sem vazamento de dados**: todo o pré-processamento fica dentro de um `Pipeline` com `ColumnTransformer`, ajustado apenas nos dados de treino de cada fold.
6. **Desbalanceamento de classes**: uso de `class_weight='balanced'`, divisão estratificada e otimização pela métrica **F1**.
7. **GridSearchCV**: busca entre kernels `linear` e `rbf`, com diferentes valores de `C` e `gamma`, usando validação cruzada estratificada de 5 folds.
8. **Avaliação**: comparação com o SVC padrão, classification report e matriz de confusão no conjunto de teste.

## 🚀 Como executar

### Google Colab

1. Abra o arquivo `SVC_GridSearchCV_AVC.ipynb` no [Google Colab](https://colab.research.google.com/).
2. Baixe o dataset no [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).
3. Na barra lateral, clique no ícone de pasta 📁 e faça upload do arquivo `healthcare-dataset-stroke-data.csv`.
4. Execute todas as células (`Ambiente de execução > Executar tudo`).

### Localmente

```bash
# Clone o repositório
git clone https://github.com/SEU-USUARIO/NOME-DO-REPOSITORIO.git
cd NOME-DO-REPOSITORIO

# Instale as dependências
pip install pandas numpy matplotlib scikit-learn jupyter

# Coloque o CSV do Kaggle na mesma pasta do notebook e abra o Jupyter
jupyter notebook SVC_GridSearchCV_AVC.ipynb
```

## 📁 Estrutura do repositório

```
├── SVC_GridSearchCV_AVC.ipynb   # Notebook com todo o projeto
└── README.md                    # Este arquivo
```

> O arquivo CSV não está incluído no repositório. Baixe-o diretamente no Kaggle.

## 📚 Tecnologias utilizadas

- Python 3
- pandas e NumPy
- scikit-learn (`SVC`, `GridSearchCV`, `Pipeline`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`)
- Matplotlib

## 👤 Autor

**Gabriel Figueiredo de Andrade**

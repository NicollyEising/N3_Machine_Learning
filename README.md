# Previsão de Preço de Imóveis com Random Forest

Este projeto utiliza um modelo de Random Forest para prever os preços de venda de imóveis. O modelo é treinado com dados de treinamento e, em seguida, faz previsões para os dados de teste.

## Requisitos

Instale as dependências utilizando o comando:

```pip install -r requirements.txt```

# Como Usar

Execute o script:

```python predict_house_prices.py```
## Certifique-se de que você tem os arquivos CSV necessários:

- `train.csv`: Contém os dados de treinamento, incluindo a coluna `SalePrice`.
- `test.csv`: Contém os dados de teste, mas sem a coluna `SalePrice`.

Esses arquivos podem ser obtidos na competição "Home Data for ML Course" do Kaggle, acessando o link abaixo:

[Home Data for ML Course - Kaggle](https://www.kaggle.com/competitions/home-data-for-ml-course/data)

O script irá treinar o modelo, avaliar seu desempenho e gerar um arquivo de submissão `submission.csv` com as previsões.

## Explicação do Código

### 1. Carregamento de Dados:
O script carrega os dados de treinamento de `train.csv` e extrai a variável dependente (`SalePrice`) e as características (`features`).

### 2. Treinamento do Modelo:
Utiliza um modelo `RandomForestRegressor` do `scikit-learn` para prever os preços dos imóveis. O modelo é treinado usando o conjunto de dados de treinamento.

### 3. Avaliação:
O modelo é avaliado no conjunto de validação com as métricas:
- **MAE** (Erro Absoluto Médio)
- **MSE** (Erro Quadrático Médio)
- **RMSE** (Raiz do Erro Quadrático Médio)
- **R²** (Coeficiente de Determinação)

### 4. Previsões e Submissão:
Após o treinamento, o modelo é ajustado ao conjunto completo de dados e realiza previsões sobre os dados de teste. Essas previsões são salvas no arquivo `submission.csv`.


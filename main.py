# Importar as bibliotecas necessárias
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# Carregar os dados
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice

# Definir as features a serem usadas
features = ['LotArea',
            'YearBuilt',
            '1stFlrSF',
            '2ndFlrSF',
            'FullBath',
            'BedroomAbvGr',
            'TotRmsAbvGrd']

# Selecionar as colunas de features
X = home_data[features]

# Dividir os dados em treinamento e validação
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Definir e treinar o modelo Random Forest
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)

# Fazer previsões com o modelo no conjunto de validação
rf_val_predictions = rf_model.predict(val_X)

# Calcular as métricas de validação
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
rf_val_mse = mean_squared_error(rf_val_predictions, val_y)
rf_val_rmse = np.sqrt(rf_val_mse)
rf_val_r2 = r2_score(val_y, rf_val_predictions)

# Exibir as métricas de validação
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
print("Validation MSE for Random Forest Model: {:,.0f}".format(rf_val_mse))
print("Validation RMSE for Random Forest Model: {:,.0f}".format(rf_val_rmse))
print("Validation R² for Random Forest Model: {:,.3f}".format(rf_val_r2))

# Agora, treinar o modelo no conjunto completo de dados
rf_model_on_full_data = RandomForestRegressor(random_state=1)
rf_model_on_full_data.fit(X, y)

# Carregar os dados de teste
test_data_path = 'test.csv'
test_data = pd.read_csv(test_data_path)

# Criar as features para os dados de teste
test_X = test_data[features]

# Fazer previsões para os dados de teste
test_preds = rf_model_on_full_data.predict(test_X)

# Criar o arquivo de submissão
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})

# Salvar o arquivo de submissão
output.to_csv('submission.csv', index=False)

print("Previsões e arquivo de submissão gerados com sucesso!")

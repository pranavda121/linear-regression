import pandas as pd
from sklearn.linear_model import LinearRegression

train_data = pd.read_csv("train.csv")
x_train = train_data[["OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF", "1stFlrSF", "FullBath", "TotRmsAbvGrd" ,"YearBuilt" ,"YearRemodAdd"]]
y_train = train_data['SalePrice']

test_data = pd.read_csv("test.csv")
x_test = test_data[ ["OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF", "1stFlrSF", "FullBath", "TotRmsAbvGrd" ,"YearBuilt" ,"YearRemodAdd" ]]
y_test = test_data['SalePrice']


model = LinearRegression()
model.fit(x_train, y_train)


y_pred = model.predict(x_test)


print(y_pred)


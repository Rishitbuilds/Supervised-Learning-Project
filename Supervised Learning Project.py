#Perform EDA and Preprocessing
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

#Define features and target
X = df[['MedInc','HouseAge','AveRooms']]
y = df['MedHouseVal']

#Inspct Data
print(df.info())
print(df.describe())

#Visualizeing the dataset
sns.pairplot(df, vars=['MedHouseVal','MedInc', 'AveRooms', "HouseAge"])
plt.show()

#Check for Mising Value
print("Missing values:\n", df.isnull().sum())

#Split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

#Make Predictions
y_pred = model.predict(X_test)   

#Model Evaluation 
mse = mean_squared_error(y_test,y_pred)         
print("Linear Regression MSE:", mse)
                      
#Perform EDA and Preprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

#Load Dataset
df_telco = pd.read_csv(r"C:\Users\User\Downloads\archive (1)\WA_Fn-UseC_-Telco-Customer-Churn.csv")
df_telco['TotalCharges'] = pd.to_numeric(df_telco['TotalCharges'])

#Inspect data 
print(df_telco.describe())
print(df_telco.info())

#Visualize the data
sns.countplot(x='Churn', data=df_telco)
plt.title("Churn Distribution")
plt.show()

#Handle missing values
df_telco.fillna(df_telco.mean(numeric_only=True), inplace=True)

#Encode Ctegorical variables
le = LabelEncoder()
df_telco['Churn'] = le.fit_transform(df_telco['Churn'])

#Define features and target
X = df_telco.drop(columns=['Churn','customerID'])
y = df_telco['Churn']

#Encode categorical features
X = pd.get_dummies(X, drop_first=True)

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Converting into standard scaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Train logistic regression model
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)

#Train K-NN model
knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
knn.fit(X_train, y_train)

#Evaluate models
log_pred = log_model.predict(X_test)
knn_pred = knn.predict(X_test)

print("\n Logistic Regression Classification report:")
print(classification_report(y_test, log_pred))

print("\n K-NN Classification report:")
print(classification_report(y_test, knn_pred))

#Confusion matrix for Logistic Regression
print("Confusion Matrix: \n", confusion_matrix(y_test, log_pred))

# Confusion Matrix for K-NN
print("\nK-NN Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))



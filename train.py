
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("loan_data.csv")

X = df[["Age", "MonthlyIncome", "CreditScore", "LoanAmount"]]
y = df["Approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy * 100, "%")

joblib.dump(model, "loan_model.pkl")

print("Model saved successfully!")

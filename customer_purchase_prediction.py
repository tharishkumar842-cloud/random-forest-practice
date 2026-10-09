import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
data = {
    "Age": [22, 25, 47, 52, 46, 56, 23, 34, 29, 41,
            35, 48, 50, 21, 27, 39, 45, 31, 24, 53],

    "Income": [25000, 30000, 70000, 90000, 65000, 95000, 28000, 50000,
               35000, 60000, 55000, 75000, 85000, 22000, 32000, 58000,
               68000, 45000, 27000, 88000],

    "Previous_Purchases": [1, 2, 8, 10, 7, 12, 1, 5, 3, 6,
                           4, 9,    11, 0, 2, 6, 8, 4, 1, 10],

    "Website_Time": [5, 8, 25, 30, 22, 35, 4, 18, 10, 20,
                     15, 28, 32, 3, 7, 19, 24, 13, 6, 29],

    "Clicked_Ad": [0, 0, 1, 1, 1, 1, 0, 1, 0, 1,
                   1, 1, 1, 0, 0, 1, 1, 0, 0, 1],

    "Purchased": [0, 0, 1, 1, 1, 1, 0, 1, 0, 1,
                  1, 1, 1, 0, 0, 1, 1, 0, 0, 1]
}
df=pd.DataFrame(data)
print("dataset",df)
X = df.drop("Purchased", axis=1)
y=df["Purchased"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
y_predict=model.predict(X_test)
accuracy=accuracy_score(y_test,y_predict)
print("accuracy",accuracy)
print("classification report",classification_report(y_test,y_predict))
print("\nconfusion matrix",confusion_matrix(y_test,y_predict))
new_customers = pd.DataFrame({
    "Age": [20, 30, 40, 55, 28],
    "Income": [20000, 45000, 60000, 90000, 35000],
    "Previous_Purchases": [0, 3, 6, 10, 2],
    "Website_Time": [3, 12, 20, 30, 8],
    "Clicked_Ad": [0, 1, 1, 1, 0]
})
predictions = model.predict(new_customers)
print(predictions)
probabilities = model.predict_proba(new_customers)
print(probabilities)
importance = model.feature_importances_
for feature, value in zip(X.columns, importance):
    print(feature, ":", value)
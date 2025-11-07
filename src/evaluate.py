from sklearn.metrics import classification_report, mean_squared_error
import joblib
from preprocess import load_and_clean

df = load_and_clean('data/Loan.csv')

# Classification
X_cls = df.drop(['LoanApproved'], axis=1)
y_cls = df['LoanApproved']
clf = joblib.load('models/loan_classifier.pkl')
y_pred_cls = clf.predict(X_cls)
print(classification_report(y_cls, y_pred_cls))

# Regression
X_reg = df.drop(['RiskScore'], axis=1)
y_reg = df['RiskScore']
reg = joblib.load('models/risk_regressor.pkl')
y_pred_reg = reg.predict(X_reg)
print("RMSE:", mean_squared_error(y_reg, y_pred_reg, squared=False))

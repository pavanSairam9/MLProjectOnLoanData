from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
import joblib
from preprocess import load_and_clean

df = load_and_clean('data/Loan.csv')

# Binary Classification
X_cls = df.drop(['LoanApproved'], axis=1)
y_cls = df['LoanApproved']
X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
joblib.dump(clf, 'models/loan_classifier.pkl')

# Regression
X_reg = df.drop(['RiskScore'], axis=1)
y_reg = df['RiskScore']
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2)

reg = GradientBoostingRegressor()
reg.fit(X_train, y_train)
joblib.dump(reg, 'models/risk_regressor.pkl')

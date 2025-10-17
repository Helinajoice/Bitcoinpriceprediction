import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
df = pd.read_csv('C:/Users/helina joice/PycharmProjects/mlskill/bitcoin (1).csv')

# Feature engineering
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['is_quarter_end'] = np.where(df['Date'].str[-5:-3].astype(int) % 3 == 0, 1, 0)
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
df = df.dropna()

# Selecting features and target
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

# Scaling features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train-test split
X_train, X_valid, Y_train, Y_valid = train_test_split(features_scaled, target, test_size=0.1, random_state=2022)

# Train the SVM model
model = SVC(kernel='poly', probability=True)
model.fit(X_train, Y_train)

# Save the model and scaler
joblib.dump(model, 'svm_bitcoin_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

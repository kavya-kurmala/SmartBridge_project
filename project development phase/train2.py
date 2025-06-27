import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# 🔹 Create dummy data that matches your model input
# Make sure values are realistic!
data = {
    'holiday': [0, 1, 0, 0, 1],
    'temp': [22.0, 18.5, 25.0, 30.0, 21.0],
    'rain': [0.0, 0.2, 0.1, 0.0, 0.3],
    'snow': [0.0, 0.0, 0.0, 1.0, 0.0],
    'weather': [1, 2, 3, 1, 4],
    'year': [2022, 2022, 2022, 2022, 2022],
    'month': [1, 2, 3, 4, 5],
    'day': [1, 15, 10, 20, 5]
}

df = pd.DataFrame(data)

# 🔹 Train the scaler
scaler = StandardScaler()
scaler.fit(df)

# 🔹 Save to scale.pkl
with open("C:/project/scale.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ New scale.pkl created and saved successfully.")


import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Dataset: Area (sq.ft), Bedrooms, House Age
X = np.array([
    [1000, 2, 5],
    [1500, 3, 10],
    [1800, 3, 8],
    [2000, 4, 5],
    [2500, 4, 2],
    [3000, 5, 1],
    [1200, 2, 12],
    [2200, 4, 4]
], dtype=float)

# Target 1: House price (in lakhs)
house_price = np.array([25, 35, 42, 50, 65, 80, 28, 58], dtype=float)

# Target 2: Number of occupants
occupants = np.array([2, 3, 4, 4, 5, 6, 2, 5], dtype=float)

# Standardize input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the SGD Regressor for house price
price_model = SGDRegressor(
    max_iter=10000,
    tol=1e-4,
    random_state=42
)
price_model.fit(X_scaled, house_price)

# Train the SGD Regressor for number of occupants
occupant_model = SGDRegressor(
    max_iter=10000,
    tol=1e-4,
    random_state=42
)
occupant_model.fit(X_scaled, occupants)

# Input details of a new house
new_house = np.array([[1600, 3, 6]])
new_house_scaled = scaler.transform(new_house)

# Predict house price and occupants
predicted_price = price_model.predict(new_house_scaled)[0]
predicted_occupants = occupant_model.predict(new_house_scaled)[0]

# Display predictions
print("House Area:", new_house[0][0], "sq.ft")
print("Number of Bedrooms:", int(new_house[0][1]))
print("House Age:", int(new_house[0][2]), "years")
print("Predicted House Price: Rs.", round(predicted_price, 2), "lakhs")
print("Predicted Number of Occupants:", max(1, round(predicted_occupants)))

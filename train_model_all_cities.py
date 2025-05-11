import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

# Collect training data from .csv files
# Add file names HERE to include them in the training data
city_files = ["geneve.csv", "lausanne.csv", "st.gallen.csv", "zurich.csv"]

# Load and merge the different datasets
dfs = []
for file in city_files:
    df = pd.read_csv(file, encoding="latin1", sep=";")
    df["source_file"] = file
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

print("Loaded rows:", len(data))
print("Columns:", data.columns)
print(data.head())

# Takes out the Zip Code 
data[['ZIP', 'City']] = data['zip_city'].str.extract(r'(\d{4})\s*(.*)')
data['ZIP'] = pd.to_numeric(data['ZIP'], errors='coerce')

# Clean and convert numbers in the numerical columns
# number_of_rooms; removes any room description besides the numerical value
data['number_of_rooms'] = data['number_of_rooms'].str.extract(r'([\d\.]+)')
data['number_of_rooms'] = pd.to_numeric(data['number_of_rooms'], errors='coerce')

# square_meters; takes just the numerical size
data['square_meters'] = data['square_meters'].str.extract(r'(\d+)')
data['square_meters'] = pd.to_numeric(data['square_meters'], errors='coerce')

# rent; only takes the price
data['rent'] = data['rent'].str.replace(r'[^\d.]', '', regex=True)
data['rent'] = pd.to_numeric(data['rent'], errors='coerce')

# Checks if essential data (zip code, number of rooms, square meters, place type and rent)
# if something is missing, the row will be skipped 
required_columns = ['ZIP', 'number_of_rooms', 'square_meters', 'place_type', 'rent']
data = data.dropna(subset=required_columns)

print("Remaining rows after cleaning:", len(data))

# Price influencing keyword detection in characteristics columns 

def detect_feature(row, keywords):
    values = [str(row['char.1']).lower(), str(row['char.2']).lower(), str(row['char.3']).lower()]
    return int(any(any(k in v for k in keywords) for v in values))

# Outdoor spaces
outdoor_keywords = ["terrace", "balcony", "garden", "patio", "loggia", "roof terrace", "outdoor"]
data['Has_Outdoor_Space'] = data.apply(lambda row: detect_feature(row, outdoor_keywords), axis=1)

# Renovated/New/Modern
modern_keywords = ["renovated", "new", "modern", "modern kitchen", "luxury"]
data['Is_Renovated_or_New'] = data.apply(lambda row: detect_feature(row, modern_keywords), axis=1)

# Parking space
parking_keywords = ["parking", "garage"]
data['Has_Parking'] = data.apply(lambda row: detect_feature(row, parking_keywords), axis=1)

# Model Training
# Put the features on the X axis against the rent on the Y axis to train a Random Forest Regressor model
X = data[['ZIP', 'number_of_rooms', 'square_meters', 'place_type', 'Is_Renovated_or_New', 'Has_Parking', 'Has_Outdoor_Space']]
y = data['rent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['place_type'])
    ],
    remainder='passthrough'
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"✅ Unified Model trained. RMSE: CHF {rmse:,.2f}")

joblib.dump(model_pipeline, "price_estimator.pkl")
print("📦 Model saved as 'price_estimator.pkl'")

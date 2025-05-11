# 🏡 Swiss Real Estate Price Estimator

This the CS Project of Group 07.10

This program provides you with a comparable rental price estimate for your apartment or house in **Geneva, Zürich, Lausanne, or St. Gallen**.

It uses machine learning based on historical rental listings to predict a price range (+- 10%) for your unit.

## 🎯 Key Features

- Step-by-step guided interface
- Enter property details:
  - Street address, ZIP, City
  - Property size and number of rooms
  - Outdoor space (balcony, terrace, roof terrace, garden, or none)
  - Renovated or recently built flag
  - Parking availability (garage, outdoor parking, or none)
- Visual map of the property location
- Rental price range (+- 10%) and estimated exact price (bold and centered)

## 🛠️ Installation

### Prerequisites

- Python 3.8+ recommended
- Pip (Python package installer)

### 1. Clone or download the project

```bash
git clone <your-repository-url>
cd CS_Project-07.10
```

### 2. Install required Python packages

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```
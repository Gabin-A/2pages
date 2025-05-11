import pandas as pd
import re

# Load raw old format file
input_file = "zurich.xlsx"   # Change this to the city you want to process
output_file = "zurich_cleaned.csv"  # Change to what the output should be (will be csv)

# Read file
df = pd.read_excel(input_file)

# Prepare cleaned dataframe
clean_df = pd.DataFrame()

# --- Extract number_of_rooms, square_meters, place_type ---
room_size_type = df['textLoadingClassname 2'].str.extract(r'(?:(\d+(?:\.\d+)?)\s*rooms?)?\s*•?\s*(\d+)\s*m²\s*•?\s*(.*)')

clean_df['number_of_rooms'] = room_size_type[0]
clean_df['square_meters'] = room_size_type[1]
clean_df['place_type'] = room_size_type[2]

# --- Extract street and zip_city ---
street_zip = df['textLoadingClassname 3'].str.extract(r'^(.*),\s*(\d{4}\s+\w+.*)$')

clean_df['street'] = street_zip[0]
clean_df['zip_city'] = street_zip[1]

# --- Copy characteristics ---
clean_df['char.1'] = df['css-8uhtka']
clean_df['char.2'] = df['css-8uhtka 2']
clean_df['char.3'] = df['css-8uhtka 3']

# --- Extract rent ---
clean_df['rent'] = df['textLoadingClassname 4'].str.replace(r'[^\d.]', '', regex=True)

# --- Extract price per m2/year ---
clean_df['p/squarem/y'] = df['textLoadingClassname 5'].str.replace(r'\s+', ' ', regex=True)

# --- Convert datatypes ---
clean_df['number_of_rooms'] = pd.to_numeric(clean_df['number_of_rooms'], errors='coerce')
clean_df['square_meters'] = pd.to_numeric(clean_df['square_meters'], errors='coerce')
clean_df['rent'] = pd.to_numeric(clean_df['rent'], errors='coerce')

# Save cleaned CSV
clean_df.to_csv(output_file, index=False)

print(f"✅ Cleaned data saved to {output_file}")

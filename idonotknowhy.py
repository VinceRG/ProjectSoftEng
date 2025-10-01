import numpy as np
import pandas as pd
import os
from openpyxl import load_workbook
import re
import calendar

# --------------------------
# Paths
# --------------------------
csv_folder  = r"D:\TRAINING MODEL\data\csv_folder"
logs_folder = r"D:\TRAINING MODEL\logs"

# Ensure logs folder exists
os.makedirs(logs_folder, exist_ok=True)

# File path for processed log
log_file = os.path.join(logs_folder, "processed_files.log")

# Final column schema
final_columns = [
    "Month_year", "Consultation_Type", "Case",
    "Under 1 Male", "Under 1 Female",
    "1-4 Male", "1-4 Female",
    "5-9 Male", "5-9 Female",
    "10-14 Male", "10-14 Female",
    "15-18 Male", "15-18 Female",
    "19-24 Male", "19-24 Female",
    "25-29 Male", "25-29 Female",
    "30-34 Male", "30-34 Female",
    "35-39 Male", "35-39 Female",
    "40-44 Male", "40-44 Female",
    "45-49 Male", "45-49 Female",
    "50-54 Male", "50-54 Female",
    "55-59 Male", "55-59 Female",
    "60-64 Male", "60-64 Female",
    "65-69 Male", "65-69 Female",
    "70 Over Male", "70 Over Female"
]

# --------------------------
# Load already processed files
# --------------------------
processed_files = set()
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        processed_files = set(line.strip() for line in f.readlines())

# --------------------------
# Scan and process only NEW CSV files
# --------------------------
for file in os.listdir(csv_folder):
    if file.endswith(".csv") and file not in processed_files:  # ✅ Skip logged files
        file_path = os.path.join(csv_folder, file)
        try:
            # --------------------------
            # STEP 1: Clean structure
            # --------------------------
            df = pd.read_csv(file_path)

            # Drop first column (extra index column)
            df = df.drop(df.columns[0], axis=1)

            # Add 2 new columns on the left
            df.insert(0, "Month_year", "")
            df.insert(1, "Consultation_Type", "")

            # Trim/pad columns to match schema
            if df.shape[1] > len(final_columns):
                df = df.iloc[:, :len(final_columns)]
            while df.shape[1] < len(final_columns):
                df[f"Extra_{df.shape[1]}"] = ""

            # Rename columns
            df.columns = final_columns

            # --------------------------
            # STEP 2: Extract Month-Year from raw text
            # --------------------------
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            match = re.search(r"MONTH AND YEAR:\s*([A-Za-z]+)\s+(\d{4})", text)
            month_year_value = ""
            if match:
                month_name = match.group(1).strip().title()
                year = match.group(2).strip()
                try:
                    month_num = list(calendar.month_name).index(month_name)
                    month_year_value = f"{year} - {month_num}"
                except ValueError:
                    pass

            if month_year_value:
                df["Month_year"] = month_year_value

            # --------------------------
            # STEP 3: Extract Consultation Type
            # --------------------------
            current_category = None
            found_categories = []

            for i, row in df.iterrows():
                for cell in row.dropna().astype(str):
                    if "TOP 10" in cell.upper():
                        last_word = re.sub(r"[^\w]", "", cell.strip().split()[-1])
                        current_category = last_word.capitalize()
                        found_categories.append((i, current_category))
                        break

                if current_category:
                    df.at[i, "Consultation_Type"] = current_category

            # --------------------------
            # STEP 4: Remove unwanted rows
            # --------------------------
            drop_indexes = []
            for i, row in df.iterrows():
                for cell in row.dropna().astype(str):
                    if "PASIG CITY CHILDREN'S HOSPITAL/PASIG CITY COVID-19 REFERRAL CENTER" in cell.upper():
                        drop_indexes.extend(range(i, i + 9))
                        break

            for i, row in df.iterrows():
                for cell in row.dropna().astype(str):
                    if "TOTAL" in cell.upper().strip():
                        drop_indexes.extend([i, i+1, i+2])
                        break

            drop_indexes = list(set(drop_indexes))
            df = df.drop(drop_indexes, errors="ignore").reset_index(drop=True)

            # --------------------------
            # STEP 5: Save and log
            # --------------------------
            df.to_csv(file_path, index=False)

            with open(log_file, "a") as f:
                f.write(file + "\n")

            print(f"✅ Processed and logged new file: {file}")

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

print("🎯 All new CSV files processed and logged.")

import os
import pandas as pd
import json

# ----------------------------
# Paths
# ----------------------------
csv_folder  = r"D:\TRAINING MODEL\data\csv_folder"
logs_folder = r"D:\TRAINING MODEL\logs"
out_folder  = r"D:\TRAINING MODEL\data\processed"

master_csv      = os.path.join(out_folder, "master_dataset.csv")
log_file        = os.path.join(logs_folder, "csv_master_log.txt")
case_dict_file  = os.path.join(logs_folder, "case_dictionary.json")

# ----------------------------
# Ensure folders exist
# ----------------------------
os.makedirs(csv_folder, exist_ok=True)
os.makedirs(logs_folder, exist_ok=True)
os.makedirs(out_folder, exist_ok=True)

# ----------------------------
# Load processed file log
# ----------------------------
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        processed_files = set(line.strip() for line in f)
else:
    processed_files = set()

# ----------------------------
# Scan for new CSV files
# ----------------------------
csv_files = [f for f in os.listdir(csv_folder) if f.lower().endswith(".csv")]
new_files = [f for f in csv_files if f not in processed_files]

if not new_files:
    print("⚠️ No new files to process.")
    exit()

print(f"📂 New files found: {new_files}")

# ----------------------------
# Load or create case dictionary
# ----------------------------
if os.path.exists(case_dict_file):
    with open(case_dict_file, "r") as f:
        case_dict = json.load(f)
else:
    case_dict = {}

# ----------------------------
# Consultation mapping
# ----------------------------
consultation_map = {
    "Consultation": 1,
    "Diagnosis": 2,
    "Mortality": 3
}

# ----------------------------
# Function to safely load CSV
# ----------------------------
def safe_load_csv(file_path):
    return pd.read_csv(file_path, engine="python", on_bad_lines="skip")

# ----------------------------
# Process new files
# ----------------------------
processed_dfs = []

for file in new_files:
    file_path = os.path.join(csv_folder, file)
    df = safe_load_csv(file_path)

    if df.empty:
        print(f"⚠️ Skipping {file}: no valid rows")
        continue

    # Align columns with master if exists
    if os.path.exists(master_csv):
        old_master = safe_load_csv(master_csv)
        master_columns = old_master.columns
        for col in master_columns:
            if col not in df.columns:
                df[col] = None
        df = df.reindex(columns=master_columns)

    # Clean 'Case' column
    if "Case" not in df.columns:
        print(f"⚠️ Skipping {file}: no 'Case' column found")
        continue

    df = df[df["Case"].notna()]
    df["Case"] = df["Case"].astype(str).str.strip()
    df = df[df["Case"] != ""]

    # Update case dictionary
    unique_cases = df["Case"].unique()
    for case in unique_cases:
        if case not in case_dict:
            case_dict[case] = len(case_dict) + 1

    df["Case"] = df["Case"].map(case_dict)

    # Encode Consultation_Type
    if "Consultation_Type" in df.columns:
        df["Consultation_Type"] = df["Consultation_Type"].map(consultation_map)

    processed_dfs.append(df)

# ----------------------------
# Append to master
# ----------------------------
if processed_dfs:
    new_data = pd.concat(processed_dfs, ignore_index=True)

    if os.path.exists(master_csv):
        old_master = safe_load_csv(master_csv)
        combined_df = pd.concat([old_master, new_data], ignore_index=True)
    else:
        combined_df = new_data

    combined_df.to_csv(master_csv, index=False)

    # Save case dictionary
    with open(case_dict_file, "w") as f:
        json.dump(case_dict, f, indent=4)

    # Update log
    with open(log_file, "a") as f:
        for file in new_files:
            f.write(file + "\n")

    print(f"✅ Appended {len(new_files)} files to master")
    print(f"📊 Total rows in master: {len(combined_df)}")
    print(f"📖 Case dictionary size: {len(case_dict)}")
else:
    print("⚠️ No valid rows to add from new files.")

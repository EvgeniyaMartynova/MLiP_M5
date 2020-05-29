import pandas as pd
import os

# merge all submission files into one
predictions_dir = "predictions"

categories_ordered = ["HOBBIES_1", "HOBBIES_2", "HOUSEHOLD_1", "HOUSEHOLD_2", "FOODS_1", "FOODS_2", "FOODS_3"]
stores_ordered = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]

files = []
for file in os.listdir(predictions_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(predictions_dir, file)
        files.append(file_path)

print(len(files))
print(files)

df = None
for store in stores_ordered:
    for category in categories_ordered:
        file_path = [c for c in files if category in c and store in c][0]

        print(file_path)
        if df is None:
            df = pd.read_csv(file_path)
        else:
            df_next = pd.read_csv(file_path)
            df = df.append(df_next)

df.to_csv(r'submission.csv', index=False)
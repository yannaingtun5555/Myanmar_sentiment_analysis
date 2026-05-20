import csv
import random

input_file = "dataset_cleaned.csv"
output_file = "dataset_cleaned.csv"

# Read all rows
with open(input_file, "r", encoding="utf-8") as infile:
    rows = list(csv.reader(infile))

# Shuffle rows
random.shuffle(rows)

# Write shuffled rows
with open(output_file, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerows(rows)

print("CSV lines shuffled successfully!")
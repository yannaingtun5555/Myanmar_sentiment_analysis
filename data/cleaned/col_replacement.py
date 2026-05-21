input_file = "test.csv"
output_file = "test1.csv"

new_second_value = "natural"

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8") as outfile:

    for line in infile:
        parts = line.strip().split(",")

        # Change second column
        if len(parts) >= 2:
            parts[1] = new_second_value

        outfile.write(",".join(parts) + "\n")

print("Done!")
input_file = "test1.csv"
output_file = "test.csv"

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8") as outfile:

    for line in infile:
        # Skip empty or whitespace-only lines
        if line.strip():
            outfile.write(line)

print("Blank lines removed!")
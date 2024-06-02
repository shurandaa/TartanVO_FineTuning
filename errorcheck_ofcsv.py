import csv

def check_csv_format(file_path):
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        line_number = 1
        errors_found = False
        for row in reader:
            # Remove any empty strings from ends, which may be due to trailing delimiters
            while row and row[-1] == '':
                print("发现空格")
                row.pop()
            # Check if the number of entries is exactly 8
            if len(row) != 8:
                print(f"Error on line {line_number}: Expected 8 entries, found {len(row)}")
                errors_found = True
            line_number += 1

        if not errors_found:
            print("No formatting errors found in the file.")

def main():
    file_path = input("data/SubT_MRS_t1/originalPredictedData.csv")
    check_csv_format(file_path)

if __name__ == "__main__":
    main()
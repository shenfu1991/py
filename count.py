import csv

def count_values(csv_file):
    value_counts = {}

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            result = row['result']
            if result in value_counts:
                value_counts[result] += 1
            else:
                value_counts[result] = 1

    return value_counts

csv_file = 'path_to_your_file.csv'  # 替换为你的CSV文件路径
result_counts = count_values(csv_file)

for result, count in result_counts.items():
    print(f'{result}: {count}')

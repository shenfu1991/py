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

csv_file = "/Users/xuanyuan/Documents/8-17-30m/ALPHAUSDT_30m_30m_processed.csv"  # 替换为你的CSV文件路径
# csv_file = '/Users/xuanyuan/py/merged_3.csv'  # 替换为你的CSV文件路径
print(csv_file)
result_counts = count_values(csv_file)

for result, count in result_counts.items():
    print(f'{result}: {count}')

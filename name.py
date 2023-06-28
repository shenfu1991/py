import os

def process_filenames(directory):
    files = os.listdir(directory)
    processed_files = [f.split('_')[0] for f in files]
    result = ', '.join(f'"{name}"' for name in processed_files)
    return result

# 调用函数，替换 'your_directory' 为你的目录路径
result = process_filenames('/Users/xuanyuan/Downloads/30m')

# 将结果写入到文件中，替换 'your_output_file.txt' 为你的输出文件路径
with open('your_output_file.txt', 'w') as output_file:
    output_file.write(result)

import os
import pandas as pd
from fuzzywuzzy import fuzz

# folder_path = "/Users/xuanyuan/Documents/30m-dum"
folder_path = "/Users/xuanyuan/Downloads/7"


print(folder_path)

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        # 创建一个空的DataFrame来存储结果
        cleaned_df = pd.DataFrame()

        for i, row in df.iterrows():
            is_duplicate = False
            for j, comp_row in df.iterrows():
                # 避免与自身进行比较
                if i != j:
                    # 比较整行数据的字符串形式
                    similarity = fuzz.ratio(' '.join(row.astype(str)), ' '.join(comp_row.astype(str)))
                    if similarity >= 80:
                        is_duplicate = True
                        break
            if not is_duplicate:
                cleaned_df = pd.concat([cleaned_df, pd.DataFrame(row).T], ignore_index=True)

        # 保存清理后的数据
        cleaned_df.to_csv(file_path, index=False)

print('done')

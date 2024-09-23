from io import StringIO
from bs4 import BeautifulSoup as bs
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

url="https://www.hko.gov.hk/tide/eCLKtext2023.html"
response = requests.get(url)
# if response.ok:
#     print("Data is ready")
    
soup = bs(response.text,'html.parser')
table = soup.find('table')
    
table_str = str(table)
table_io = StringIO(table_str)
df = pd.read_html(table_io,header=1)[0]

df=df.values

# 合并前三列
merged_column = np.array([''.join(map(str, row[:3])) for row in df]).reshape(-1, 1)

# 保持其余列不变
remaining_columns = df[:, 3:]

# 合并结果
result = np.hstack((merged_column, remaining_columns))

np.set_printoptions(suppress=True)

def convert_to_datetime(date_str):
    # 确保输入是字符串
    date_str = str(date_str)
    # 解析字符串
    parts = date_str.split('.')
    if len(parts) == 4:
        day, month, year, hour = parts
        # 将其转换为 datetime 对象
        dt = datetime.strptime(f"{day}.{month}.{year}.{hour}", "%d.%m.%Y.%H")
        # 返回格式化后的字符串
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        # 如果格式不正确，返回原始字符串或处理错误
        return date_str

# 转换第一列
converted_column = np.array([convert_to_datetime(row[0]) for row in df]).reshape(-1, 1)

# 保持其余列不变
remaining_columns = df[:, 1:]

# 合并结果
df = np.hstack((converted_column, remaining_columns))
df= np.delete(df, 2, axis=1)

df = pd.DataFrame(df)
df['Date'] = df.apply(lambda row: f"2023-{int(float(row[0])):02d}-{int(float(row[1])):02d}", axis=1)

# 删除原来的前两列
df = df.drop(columns=[0, 1])

column_names = df.columns.tolist()
df.columns = ['waterlevel0','无效1', 'waterlevel1', '无效2','waterlevel2','无效3','waterlevel3','DATE']
df_new = df[['waterlevel0','waterlevel1','waterlevel2','waterlevel3','DATE']]
plt.figure(figsize=(10, 6))

# 绘制折线图
plt.plot(df_new['DATE'], df_new['waterlevel0'], label='Water Level 0', marker='o')
plt.plot(df_new['DATE'], df_new['waterlevel1'], label='Water Level 1', marker='o')
plt.plot(df_new['DATE'], df_new['waterlevel2'], label='Water Level 2', marker='o')
plt.plot(df_new['DATE'], df_new['waterlevel3'], label='Water Level 3', marker='o')

# 添加标题和标签
plt.title('Water Levels Over Time')
plt.xlabel('Date')
plt.ylabel('Water Level')
plt.xticks(rotation=45)  # 旋转日期标签以便更好地显示
plt.legend()  # 显示图例
plt.grid()  # 添加网格

# 显示图形
plt.tight_layout()  # 自动调整布局
plt.show()
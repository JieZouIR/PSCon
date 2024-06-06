import json

# 读取 JSON 文件
with open('../dataset/conversation_cn.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 遍历每条数据，并将 "conversation" 字段转换为字符串类型
for item in data:
    conversation_str = json.dumps(item['conversation'], ensure_ascii=False)
    item['conversation'] = conversation_str

# 将处理后的数据写回到 JSON 文件
with open('../dataset/conversation_cn_croissant_data.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=2, ensure_ascii=False)

print("Conversation fields converted to string successfully.")


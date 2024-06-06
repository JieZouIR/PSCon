import pandas as pd
import json

# 读取JSON文件
with open('conversation_en_processed4.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 初始化一个空的DataFrame列表
df_list = []

# 遍历每个对话
for conversation in data:
    conv_id = conversation['conv_id']
    for msg in conversation['conversation']:
        if msg is None:
            continue
        # 创建一个字典，包含所有消息的信息
        msg_info = {
            'conv_id': conv_id,
            'msg_id': msg['msg_id'],
            'turn_id': msg['turn_id'],
            'role': msg['role'],
            'content': msg['content'],
            'action': msg.get('action', ''),  # 使用get方法以避免KeyError
            'clarifying_attribute': msg.get('clarifying_attribute', []),
            'keywords': msg.get('keywords', []),
            'intent': msg.get('intent', '')
        }
        # 将消息信息添加到列表中
        df_list.append(msg_info)

# 将列表转换为DataFrame
df = pd.DataFrame(df_list)

# 打印DataFrame
print(df)
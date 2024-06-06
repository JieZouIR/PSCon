import json


with open('conversation_cn_0605.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    print(len(data))

# with open('0401/getEnData/conv.json', 'r', encoding='utf-8') as file:
#     data = json.load(file)
#     print(len(data))

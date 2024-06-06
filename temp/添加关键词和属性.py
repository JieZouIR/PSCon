import json

id2keywords = {}
id2clarify = {}
with open('conversation_en_0605.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    for item in data:
        for message in item['conversation']:
            if "keywords" in message:
                id2keywords[message['msg_id']] = message['keywords']
            if "clarifying_attribute" in message:
                id2clarify[message['msg_id']] = message['clarifying_attribute']

with open('../dataset/conversation_en.json', 'r', encoding='utf-8') as fw:
    data = json.load(fw)
    for item in data:
        for message in item['conversation']:
            if message['msg_id'] in id2keywords:
                message['keywords'] = id2keywords[message['msg_id']]
            elif message['msg_id'] in id2clarify:
                message['clarifying_attribute'] = id2clarify[message['msg_id']]

with open('../dataset/conversation_en.json', 'w', encoding='utf-8') as fw:
    fw.write(json.dumps(data, indent=2, ensure_ascii=False))


# with open('conversation_cn_0605.json', 'r', encoding='utf-8') as fr:
#     data = json.load(fr)
#     for item in data:
#         for message in item['conversation']:
#             if "clarifying_attribute" in message and len(message["clarifying_attribute"]) > 0:
#                 message['clarifying_attribute'] = [i.strip(',') for i in message['clarifying_attribute']]
#
# with open('conversation_cn_0605.json', 'w', encoding='utf-8') as fw:
#     fw.write(json.dumps(data, indent=2, ensure_ascii=False))
#
# with open('conversation_cn_0605.json', 'r', encoding='utf-8') as fr:
#     data = json.load(fr)
#     for item in data:
#         for message in item['conversation']:
#             if "clarifying_attribute" in message and len(message["clarifying_attribute"]) > 0:
#                 print(message["clarifying_attribute"])

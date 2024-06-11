import json
import os

with open('conversation_en.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    sentences = {}
    for item in data:
        for message in item['conversation']:
            if message['content'] not in sentences:
                sentences[message['content']] = 1
            else:
                sentences[message['content']] += 1

# 按照出现的数量从大到小排序
sorted_sentences = sorted(sentences.items(), key=lambda x: x[1], reverse=True)

# print(sorted_sentences)

for item in sorted_sentences:
    if item[1] > 10:
        print(item)

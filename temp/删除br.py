import json

with open('../temp/conversation_en_0611.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    for item in data:
        for message in item['conversation']:
            sentence = message['content']
            if "<br>" in sentence:
                message['content'] = sentence.replace("<br>", "")

with open('conversation_en.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(data, ensure_ascii=False, indent=2))

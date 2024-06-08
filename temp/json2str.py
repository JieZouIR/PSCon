import json

with open('conversation_en.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    for item in data:
        item['conversation'] = str(item['conversation'])

with open('conversation_en_croissant_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

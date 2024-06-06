import json

with open('conversation_en_processed5.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    lens = {
        "conversation": 0,
        "keywords": 0,
        "clarifying_attribute": 0,
        "search_results": 0,
        "recommended_products": 0,
        "user_rating": 0
    }
    for item in data:
        lens['conversation'] = max(lens['conversation'], len(item['conversation']))
        for message in item['conversation']:
            if message is None:
                continue
            if "keywords" in message:
                lens['keywords'] = max(lens['keywords'], len(message['keywords']))
            if "clarifying_attribute" in message:
                lens['clarifying_attribute'] = max(lens['clarifying_attribute'], len(message['clarifying_attribute']))
            if "search_results" in message:
                lens['search_results'] = max(lens['search_results'], len(message['search_results']))
            if "recommended_products" in message:
                lens['recommended_products'] = max(len(message['recommended_products']), lens['recommended_products'])
            if "user_rating" in message:
                lens['user_rating'] = max(len(message['user_rating']), lens['user_rating'])

    for item in data:
        if len(item['conversation']) != lens['conversation']:
            print(000)
        for message in item['conversation']:
            if message is None:
                continue
            if "keywords" in message and len(message['keywords']) != lens['keywords']:
                print(111)
                message['keywords'].extend([None] * (lens['keywords'] - len(message['keywords'])))
            elif "keywords" not in message:
                message['keywords'] = ([None] * (lens['keywords']))
            if "clarifying_attribute" in message and len(message['clarifying_attribute']) != lens['clarifying_attribute']:
                print(222)
                message['clarifying_attribute'].extend([None] * (lens['clarifying_attribute']-len(message['clarifying_attribute'])))
            elif "clarifying_attribute" not in message:
                message['clarifying_attribute'] = ([None] * (lens['clarifying_attribute']))
            if "search_results" in message and len(message['search_results']) != lens['search_results']:
                print(333)
                message['search_results'].extend([None] * (lens['search_results']-len(message['search_results'])))
            elif "search_results" not in message:
                message['search_results'] = ([None] * (lens['search_results']))
            if "recommended_products" in message and len(message['recommended_products']) != lens['recommended_products']:
                print(444)
                message['recommended_products'].extend([None] * (lens['recommended_products']-len(message['recommended_products'])))
            elif "recommended_products" not in message:
                message['recommended_products'] = ([None] * (lens['recommended_products']))
            if "user_rating" in message and len(message['user_rating']) != lens['user_rating']:
                print(555)
                message['user_rating'].extend([None] * (lens['user_rating']-len(message['user_rating'])))
            elif "user_rating" not in message:
                message['user_rating'] = ([None] * (lens['user_rating']))

# with open('conversation_en_processed5.json', 'w', encoding='utf-8') as fw:
#     fw.write(json.dumps(data, indent=4, ensure_ascii=False))

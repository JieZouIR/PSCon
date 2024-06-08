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



# 定义一个函数，读取文件并返回文件内容
def read_files_in_folder(folder_path):
    data = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            if filename.startswith('Thank you'):
                data["Thank you."] = file.read().splitlines()
            else:
                data[filename] = file.read().splitlines()
    return data


# 定义一个函数，替换对话中匹配到的原句为相应的相似句子
def replace_sentence(sentence, file_data, matched_sentences):
    for idx, orig_sentence in enumerate(matched_sentences):
        if orig_sentence == "Thank you":
            orig_sentence = "Thank you."
        if orig_sentence in sentence:
            return file_data[orig_sentence][matched_sentences[orig_sentence]], orig_sentence, True
    return sentence, sentence, False


# 读取tmp文件夹下的所有文件
folder_path = 'tmp'
file_data = read_files_in_folder(folder_path)


# 用来存储已经匹配的句子，以及已经替换的相似句子的索引
matched_sentences = {}
for orig_sentence in file_data.keys():
    matched_sentences[orig_sentence] = 0

with open('conversation_en.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    sentences = {}
    for item in data:
        for message in item['conversation']:
            sentence = message['content']
            message['content'], orig_sentence, matched = replace_sentence(sentence, file_data, matched_sentences)
            if matched:
                print(f"原句：{sentence}, 修改后:{message['content']}")
                matched_sentences[orig_sentence] += 1
                matched_sentences[orig_sentence] %= len(file_data[orig_sentence])

with open('conversation_en_after.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

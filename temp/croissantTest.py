import mlcroissant as mlc

import json

ds = mlc.Dataset("../dataset/croissant-conversation-en.json")
metadata = ds.metadata.to_json()
print(f"{metadata['name']}: {metadata['description']}")
i = 1
for x in ds.records(record_set="default"):
    i += 1
    print(x)
    if i == 10:
        exit(000)

import json
from collections import defaultdict
from pandas import DataFrame


def load_json(content):
    json_data = []

    for line in content:
        json_data.append(json.loads(line))
    
    return json_data

def get_training_set(training_file, n):
    with open(training_file) as file:
        content = file.readlines()
    json_data = load_json(content)
    # create_ngrams(json_data, n)
    build_dataframe(json_data)

def build_dataframe(json_data):
    rows = []
    for obj in json_data:
        rows.append({'text' : obj['text'], 'class' : obj['lang']})
    data_frame = DataFrame(rows)
    return data_frame

def create_dictionary(json_data):
    language_dict = defaultdict(list)
    for obj in json_data:
        language_dict[obj['lang']].append(obj['text'])
    return language_dict

def create_ngrams(json_data, n):
    language_dict = create_dictionary(json_data)
    numthreads = 20
    input_languages = []

    # Store the languages detected.
    for lang in language_dict:
        input_languages.append(lang)

get_training_set("train.json", 3)



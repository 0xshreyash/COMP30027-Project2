import json
from html.parser import HTMLParser
from collections import defaultdict
from pandas import DataFrame
from string import digits
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def load_json(content):
    json_data = []

    for line in content:
        json_data.append(json.loads(line))

    return json_data

def train_simple_classifier(training_file):
    with open(training_file) as file:
        content = file.readlines()
    json_data = load_json(content)
    clean_json_data = clean_data(json_data)
    # create_ngrams(json_data, n)
    data_frame = build_dataframe(clean_json_data)
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(1, 4))),
        ('classifier', MultinomialNB())
    ])
    pipeline.fit(data_frame['text'].values, data_frame['class'].values)
    return pipeline

def clean_data(json_data):
    clean_json_data = json_data
    html_parser = HTMLParser()
    remove_digits = str.maketrans('', '', digits)
    count = 0
    for obj in clean_json_data:
        obj['text'] = html_parser.unescape(obj['text'])
        obj['text'] = obj['text'].translate(remove_digits)
        obj['text'] = ' '.join(obj['text'].split())
        if count == 0:
            print(obj['text'])
        count += 1

    return clean_json_data

def build_dataframe(json_data):
    rows = []
    index = []
    for obj in json_data:
        rows.append({'text' : obj['text'], 'class' : obj['lang']})
        index.append(obj['src'])
    data_frame = DataFrame(rows, index=index)
    # Try shuffling to increase accuracy.
    # print(data_frame)
    return data_frame

def create_dictionary(json_data):
    language_dict = defaultdict(list)
    for obj in json_data:
        language_dict[obj['lang']].append(obj['text'])
    return language_dict

def create_ngrams(json_data, n):
    language_dict = create_dictionary(json_data)
    input_languages = []
    # Store the languages detected.
    for lang in language_dict:
        input_languages.append(lang)

def main():
    pipeline = train_simple_classifier("train.json")
    with open("dev.json") as file:
        dev_content = file.readlines()
    json_dev_data = load_json(dev_content)
    to_predict = []
    answers = []
    for obj in json_dev_data:
        to_predict.append(obj['text'])
        answers.append(obj['lang'])
    predicted_probabilities = pipeline.predict_proba(to_predict)
    predictions = pipeline.predict(to_predict)
    final_predictions = []
    for i in range(len(predictions)):
        if max(predicted_probabilities[i]) < 0.60:
            print(max(predicted_probabilities[i]))
            print(str(predictions[i]) + " " + str(answers[i]))
            final_predictions.append('unk')
        else:
            final_predictions.append(predictions[i])

    count = 0
    total = 0
    unks_wrong = 0
    for i in range(len(answers)):
        if predictions[i] == answers[i]:
            count += 1
        else:
            print("Predicted:" + str(predictions[i]) + " Real:" + str(answers[i]))
            print(str(predicted_probabilities[i]))
            if answers[i] == 'unk':
                unks_wrong += 1
        total += 1
    '''for i in range(len(answers)):
        if answers[i] == 'unk':
            print(str(max(predicted_probabilities[i])))
    '''
    print(str(count) + " " + str(total) + " "  + str(unks_wrong))
    
main()



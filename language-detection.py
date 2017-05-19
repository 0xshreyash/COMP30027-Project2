import json
from string import digits, punctuation
from html.parser import HTMLParser
from collections import defaultdict
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import re

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
        ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(2, 4))),
        ('classifier', MultinomialNB())
    ])
    pipeline.fit(data_frame['text'].values, data_frame['class'].values)
    return pipeline

def train_svm_classifier(training_file):
    with open(training_file) as file:
        content = file.readlines()
    print("Going to get data")
    json_data = load_json(content)
    print("Data got")
    clean_json_data = clean_data(json_data)
    # create_ngrams(json_data, n)
    data_frame = build_dataframe(clean_json_data)
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(analyzer='char_wb', ngram_range=(1, 3))),
        ('classifier', LinearSVC())
    ])
    pipeline.fit(data_frame['text'].values, data_frame['class'].values)
    return pipeline


def clean_data(json_data):
    clean_json_data = json_data
    html_parser = HTMLParser()
    remove_digits = str.maketrans('', '', digits)
    remove_punctuation = re.compile('[%s]' % re.escape(punctuation))
    remove_twitter_specific_stuff = re.compile('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)')
    # remove_punctutation = re.compile(punctuation())
    count = 0
    prev_source = ""
    for obj in clean_json_data:
        obj['text'] = html_parser.unescape(obj['text'])
        obj['text'] = obj['text'].translate(remove_digits)
        # obj['text'] = remove_punctuation.sub('', obj['text'])
        # obj['text'] = ' '.join(remove_twitter_specific_stuff.sub(" ", obj['text']).split())
        obj['text'] = ' '.join(obj['text'].split())
        if prev_source != obj['src']:
            print(obj['src'])
            print(obj['text'])
        prev_source = obj['src']
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

def svm_system():
    pipeline = train_svm_classifier("train.json")
    print("Training done!")
    with open("test.json") as file:
        dev_content = file.readlines()
    json_dev_data = load_json(dev_content)
    to_predict = []
    answers = []
    for obj in json_dev_data:
        to_predict.append(obj['text'])
        answers.append(obj['lang'])
    print("Fine here")
    # predicted_probabilities = pipeline.predict_proba(to_predict)
    predictions = pipeline.predict(to_predict)
    score = pipeline.decision_function(to_predict)
    '''for i in range(len(score)):
        if(answers[i] == 'unk'):
            print(score[i])'''
    final_predictions = []
    for i in range(len(predictions)):
        if max(score[i]) < 0:
            print(max(score[i]))
            print(str(predictions[i]) + " " + str(answers[i]))
            final_predictions.append('unk')
        else:
            final_predictions.append(predictions[i])

    count = 0
    total = 0
    unks_wrong = 0
    for i in range(len(answers)):
        if final_predictions[i] == answers[i]:
            count += 1
        else:
            # print("Predicted:" + str(final_predictions[i]) + " Real:" + str(answers[i]))
            # print(str(predicted_probabilities[i]))
            if answers[i] == 'unk':
                unks_wrong += 1

        total += 1
    '''for i in range(len(answers)):
        if answers[i] == 'unk':
            print(str(max(predicted_probabilities[i])))
    '''
    print(str(count) + " " + str(total) + " "  + str(unks_wrong))

def naive_bayes_system():
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
        if max(predicted_probabilities[i]) < 0.30:
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
            # print("Predicted:" + str(final_predictions[i]) + " Real:" + str(answers[i]))
            # print(str(predicted_probabilities[i]))
            if answers[i] == 'unk':
                unks_wrong += 1

        total += 1
    print(str(count) + " " + str(total) + " "  + str(unks_wrong))
    
def main():
    svm_system()

main()


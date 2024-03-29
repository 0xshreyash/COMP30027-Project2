'''
Author: Shreyash Patodia
Student ID: 767336
Login: spatodia
Subject: COMP30027 Machine Learning
Project-2 Language Identification
'''
import json
import re
from html.parser import HTMLParser
from string import digits, punctuation
import numpy as np
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from pandas import DataFrame

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from fuzzywuzzy import fuzz
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
import seaborn as sn
import csv


##############################################################################################
                                  # Common #
def load_json(content):
# Loads json into an array
    json_data = []
    for line in content:
        json_data.append(json.loads(line))
    return json_data

def classify_locations(clean_json_data):
# Gets location data using geopy
    count = 0
    for obj in clean_json_data:
        if 'location' in obj:
            if fuzz.ratio(obj['location'], 'not given') > 80:
                # print('Hello')
                obj['location'] = ''
            elif fuzz.ratio(obj['location'], 'None') > 80:
                obj['location'] = ''
            else:
                try:
                    geolocator = Nominatim(timeout=100)
                    obj['location'] = geolocator.geocode(obj['location'])
                except GeocoderTimedOut:
                    obj['location'] = ''
                    print("Error: geocode failed on input %s with message:"
                          %(obj['location']))
                except ConnectionResetError:
                    geolocator = Nominatim(timeout=100)
                    obj['location'] = ''
                    continue
        else:
            obj['location'] = ''
        count += 1
        print(count)
        print(obj['location'])
    return clean_json_data

def strip_links(text):
# Strips away links from document
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

def strip_all_entities(text):
# Takes away hashtags and usernames
    entity_prefixes = ['@','#']
    '''for separator in punctuation:
        if separator not in entity_prefixes and separator != "'":
            text = text.replace(separator,' ')
    '''
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes and word != 'rt':
                words.append(word)
    return ' '.join(words)

# Used for FeatureUnion, not part of the final submission.
class TextLocationExtractor(BaseEstimator, TransformerMixin):
    """Extract the subject & body from a usenet post in a single pass.

    Takes a sequence of strings and produces a dict of sequences.  Keys are
    `subject` and `body`.
    """
    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        features = np.recarray(shape=(len(data_frame)),
                               dtype=[('text', object), ('location', object)])
        i = 0 
        for row in data_frame:
            features['text'][i] = row[0]
            features['location'][i] = row[1]
            i += 1
        return features

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
            self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

#############################################################################################
                                # Naive Bayes #

def train_bayes_classifier(training_file):
# Trains the naive bayes classifier.
    with open(training_file) as file:
        content = file.readlines()
    json_data = load_json(content)
    clean_json_data = naive_clean_data(json_data)
    # create_ngrams(json_data, n)
    data_frame = naive_build_dataframe(clean_json_data)
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(analyzer='char_wb', ngram_range=(2, 4))),
        ('classifier', MultinomialNB())
    ])
    pipeline.fit(data_frame['text'].values, data_frame['class'].values)
    return pipeline

def naive_bayes_system():
    print("Running naive bayes classifer..")
# Call this function to run the whole naive bayes system.
    pipeline = train_bayes_classifier("train.json")
    print("Training for naive bayes classifier done.")
    with open("dev.json") as file:
        dev_content = file.readlines()
    json_dev_data = load_json(dev_content)
    clean_dev_data = naive_clean_data(json_dev_data)
    to_predict = []
    answers = []
    doc_ids = []
    num = 0
    for obj in clean_dev_data:
        doc_ids.append("text{:04}".format(num))
        to_predict.append(obj['text'])
        answers.append(obj['lang'])
        num = num + 1
    predicted_probabilities = pipeline.predict_proba(to_predict)
    predictions = pipeline.predict(to_predict)
    final_predictions = []
    for i in range(len(predictions)):
        if max(predicted_probabilities[i]) < 0.30:
            # print(max(predicted_probabilities[i]))
            # print(str(predictions[i]) + " " + str(answers[i]))
            final_predictions.append('unk')
        else:
            final_predictions.append(predictions[i])
    
    count = 0
    total = 0
    unks_wrong = 0
    resultFile = open("naive-dev.csv", "w")
    resultFile.write("docid,lang\n")
    for i in range(len(final_predictions)):
        resultFile.write(str(doc_ids[i]) + "," + str(final_predictions[i]) + "\n")
    for i in range(len(answers)):
        if final_predictions[i] == answers[i]:
            count += 1
        else:
            # print("Predicted:" + str(final_predictions[i]) + " Real:" + str(answers[i]))
            # print(str(predicted_probabilities[i]))
            if answers[i] == 'unk':
                unks_wrong += 1
        total += 1
    print("Dev data testing done! Printing stats:")
    print("Correct: " + str(count) + " Total: " + str(total) + 
    " Number of unks wrong: "  + str(unks_wrong))
    print("Now running on test data..")
    with open("project2/test.json") as file:
        dev_content = file.readlines()
    json_dev_data = load_json(dev_content)
    clean_dev_data = naive_clean_data(json_dev_data)
    to_predict = []
    answers = []
    doc_ids = []
    for obj in clean_dev_data:
        to_predict.append(obj['text'])
        doc_ids.append(obj['id'])
        # answers.append(obj['lang'])
    predicted_probabilities = pipeline.predict_proba(to_predict)
    predictions = pipeline.predict(to_predict)
    final_predictions = []
    for i in range(len(predictions)):
        if max(predicted_probabilities[i]) < 0.60:
            # print(max(predicted_probabilities[i]))
            # print(str(predictions[i]) + " " + str(answers[i]))
            final_predictions.append('unk')
        else:
            final_predictions.append(predictions[i])
    resultFile = open("naive-test.csv", "w")
    resultFile.write("docid,lang\n")
    for i in range(len(final_predictions)):
        resultFile.write(str(doc_ids[i]) + "," + str(final_predictions[i]) + "\n")
    print("Testing data run done..")
    print("=================End of Naive Bayes System========================")


def naive_build_dataframe(json_data):
    rows = []
    index = []
    for obj in json_data:
        if obj['src'] == 'twitter':
            if 'location' not in obj:
                obj['location'] = ''
            elif fuzz.ratio(obj['location'], 'not given') > 0.80:
                obj['location'] = ''
            elif fuzz.ratio(obj['location'], 'None') > 0.70:
                obj['location'] = ''
            rows.append({'text' : obj['text'], 'location': obj['location'], 'class' : obj['lang']})
            index.append(obj['src'])
    data_frame = DataFrame(rows, index=index)
    # Try shuffling to increase accuracy.
    # print(data_frame)
    return data_frame


def naive_clean_data(json_data):
    clean_json_data = json_data
    html_parser = HTMLParser()
    remove_digits = str.maketrans('', '', digits)
    # remove_punctuation = re.compile('[%s]' % re.escape(punctuation))
    # remove_twitter_specific_stuff = re.compile('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)')
    # remove_punctutation = re.compile(punctuation())
    for obj in clean_json_data:
        obj['text'] = html_parser.unescape(obj['text'])
        obj['text'] = obj['text'].translate(remove_digits)
        # obj['text'] = remove_punctuation.sub('', obj['text'])
        # obj['text'] = ' '.join(remove_twitter_specific_stuff.sub(" ", obj['text']).split())

    return clean_json_data

#############################################################################################
                            # Support Vector Machine #

def svm_training_clean_data(json_data):
    clean_json_data = json_data
    html_parser = HTMLParser()
    remove_digits = str.maketrans('', '', digits)
    count = 0
    prev_source = ''
    for obj in clean_json_data:
        count += 1
        # print('Before :' + obj['text'])
        obj['text'] = obj['text'].lower()
        obj['text'] = html_parser.unescape(obj['text'])
        obj['text'] = obj['text'].translate(remove_digits)
        obj['text'] = ' '.join(obj['text'].split())
        obj['text'] = strip_links(obj['text'])
        # obj['text'] = strip_all_entities(obj['text'])
        # print('After :' + obj['text'])
    return clean_json_data

def svm_dev_clean_data(json_data):
    clean_json_data = json_data
    html_parser = HTMLParser()
    remove_digits = str.maketrans('', '', digits)
    count = 0
    for obj in clean_json_data:
        # print('Before :' + obj['text'])
        obj['text'] = obj['text'].lower()
        obj['text'] = html_parser.unescape(obj['text'])
        obj['text'] = obj['text'].translate(remove_digits)
        obj['text'] = ' '.join(obj['text'].split())
        obj['text'] = strip_links(obj['text'])
        # obj['text'] = strip_all_entities(obj['text'])
        # print('After :' + obj['text'])
        count += 1
    return clean_json_data

def svm_build_dataframe(json_data):
    rows = []
    index = []
    for obj in json_data:
        if 'location' not in obj:
            obj['location'] = ''
        elif fuzz.ratio(obj['location'], 'not given') > 0.80:
            obj['location'] = ''
        elif fuzz.ratio(obj['location'], 'None') > 0.70:
            obj['location'] = ''
        rows.append({'text' : obj['text'], 'location': obj['location'], 
        'class' : obj['lang']})
        index.append(obj['src'])
    data_frame = DataFrame(rows, index=index)
    # Try shuffling to increase accuracy.
    # print(data_frame)
    return data_frame

def train_svm_classifier(training_file):
    with open(training_file) as file:
        content = file.readlines()
    # print("Going to get data")
    json_data = load_json(content)
    # print("Data got")
    clean_json_data = svm_training_clean_data(json_data)
    # print(clean_json_data)
    # clean_json_data = classify_locations(clean_json_data)
    # create_ngrams(json_data, n)
    data_frame = svm_build_dataframe(clean_json_data)
    
    pipeline = Pipeline([
       ('weighed_vectorizer', TfidfVectorizer(analyzer='char', ngram_range=(2, 4))),
       ('classifier', LinearSVC(C=0.9, class_weight='balanced'))
    ])

    # X = build_X(data_frame)
    # X = data_frame[['text', 'location']].values

    pipeline.fit(data_frame['text'].values, data_frame['class'].values)
    # pipeline.fit(data_frame['location'].values, data_frame['class'].values)
    return pipeline

def build_X(data_frame):
    rows = []
    for row in data_frame:
        rows.append({'text' : row[0], 'location': row[1].lower()})
    new_data_frame = DataFrame(rows)
    return new_data_frame


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
    print("Now running the Support Vector Machine. This may take a while..")
    pipeline = train_svm_classifier("train.json")
    print("Training for SVM done!")
    with open("dev.json") as file:
        dev_content = file.readlines()
    json_dev_data = load_json(dev_content)
    clean_dev_data = svm_dev_clean_data(json_dev_data)
    to_predict = []
    answers = []
    doc_ids = []
    num = 0
    for obj in clean_dev_data:
        doc_ids.append("text{:04}".format(num))
        to_predict.append(obj['text'])
        answers.append(obj['lang'])
        num += 1
    # predicted_probabilities = pipeline.predict_proba(to_predict)
    predictions = pipeline.predict(to_predict)
    score = pipeline.decision_function(to_predict)
    final_predictions = []
    for i in range(len(predictions)):
        if max(score[i]) < -0.36:
            # print(max(score[i]))
            # print(str(predictions[i]) + " " + str(answers[i]))
            final_predictions.append('unk')
        else:
            final_predictions.append(predictions[i])
    resultFile = open("svm-dev.csv", "w")
    resultFile.write("docid,lang\n")
    for i in range(len(final_predictions)):
        resultFile.write(doc_ids[i] + "," + final_predictions[i] + "\n")
    print("SVM on dev data done! Moving on to test data..")
    count = 0
    total = 0
    unks_wrong = 0
    max_score_unk = -1000;
    for i in range(len(answers)):
        
        if final_predictions[i] == answers[i]:
            count += 1
        else:
        
            # print(max(score[i]))
            # print("Predicted:" + str(final_predictions[i]) + " Real:" + str(answers[i]))
            # print(str(predicted_probabilities[i]))
            if answers[i] == 'unk':
                if(max(score[i]) > max_score_unk):
                    max_score_unk = max(score[i])
                unks_wrong += 1
        total += 1
    print(max_score_unk)
    print("Correct: " + str(count) + " Total: " + str(total) + " Unks wrong: "  + str(unks_wrong))
    with open("project2/test.json") as file:
        dev_content = file.readlines()
    json_dev_data = load_json(dev_content)
    clean_dev_data = svm_dev_clean_data(json_dev_data)
    to_predict = []
    answers = []
    doc_ids = []
    for obj in clean_dev_data:
        doc_ids.append(obj['id'])
        to_predict.append(obj['text'])
        # answers.append(obj['lang'])
    # predicted_probabilities = pipeline.predict_proba(to_predict)
    predictions = pipeline.predict(to_predict)
    score = pipeline.decision_function(to_predict)
    final_predictions = []
    for i in range(len(predictions)):
        if max(score[i]) < -0.36:
            # print(max(score[i]))
            # print(str(predictions[i]) + " " + str(answers[i]))
            final_predictions.append('unk')
        else:
            final_predictions.append(predictions[i])
    resultFile = open("svm-test.csv", "w")
    resultFile.write("docid,lang\n")
    for i in range(len(final_predictions)):
        resultFile.write(doc_ids[i] + "," + final_predictions[i] + "\n")
    print("Predictions on testing data done!")
    print("=================End of SVM Classifier========================")
    
#############################################################################################
                             # Logistic Regression #

def lr_training_clean_data(json_data):
    # Clean training data
    clean_json_data = json_data
    html_parser = HTMLParser()
    remove_digits = str.maketrans('', '', digits)
    count = 0
    for obj in clean_json_data:
        if obj['src'] == 'twitter':
            # print('Before :' + obj['text'])
            obj['text'] = obj['text'].lower()
            obj['text'] = html_parser.unescape(obj['text'])
            obj['text'] = obj['text'].translate(remove_digits)
            obj['text'] = ' '.join(obj['text'].split())
            # obj['text'] = strip_links(obj['text'])
            # obj['text'] = strip_all_entities(obj['text'])
            # print('After :' + obj['text'])
        else:
            clean_json_data.remove(obj)
        count += 1
    return clean_json_data

def lr_dev_clean_data(json_data):
    # Clean development/test data 
    clean_json_data = json_data
    html_parser = HTMLParser()
    remove_digits = str.maketrans('', '', digits)
    count = 0
    for obj in clean_json_data:
        # print('Before :' + obj['text'])
        obj['text'] = obj['text'].lower()
        obj['text'] = html_parser.unescape(obj['text'])
        obj['text'] = obj['text'].translate(remove_digits)
        obj['text'] = ' '.join(obj['text'].split())
        # obj['text'] = strip_links(obj['text'])
        # obj['text'] = strip_all_entities(obj['text'])
        # print('After :' + obj['text'])
 
        count += 1
    return clean_json_data

def lr_build_dataframe(json_data):
    rows = []
    index = []
    for obj in json_data:
        if obj['src'] == 'twitter':
            if 'location' not in obj:
                obj['location'] = ''
            elif fuzz.ratio(obj['location'], 'not given') > 0.80:
                obj['location'] = ''
            elif fuzz.ratio(obj['location'], 'None') > 0.70:
                obj['location'] = ''
            rows.append({'text' : obj['text'], 'location': obj['location'],
            'class' : obj['lang']})
            index.append(obj['src'])
    data_frame = DataFrame(rows, index=index)
    # Try shuffling to increase accuracy.
    # print(data_frame)
    return data_frame

def train_lr_classifier(training_file):
    with open(training_file) as file:
        content = file.readlines()
    # print("Going to get data")
    json_data = load_json(content)
    # print("Data got")
    clean_json_data = lr_training_clean_data(json_data)
    # clean_json_data = classify_locations(clean_json_data)
    # create_ngrams(json_data, n)
    data_frame = lr_build_dataframe(clean_json_data)
    pipeline = Pipeline([
        # ('vectorizer', CountVectorizer(analyzer='char_wb', ngram_range=(2, 4))),
        ('weighed_vectorizer', TfidfVectorizer(analyzer='char', ngram_range=(2, 4))),
        ('classifier', LogisticRegression())
    ])
    # X = build_X(data_frame)
    # X = data_frame[['text', 'location']].values
    pipeline.fit(data_frame['text'].values, data_frame['class'].values)
    # pipeline.fit(data_frame['location'].values, data_frame['class'].values)
    return pipeline

def lr_system():
    print("Running logistic regression system now: ")
    pipeline = train_lr_classifier("train.json")
    print("Training for LR system done..")
    with open("dev.json") as file:
        dev_content = file.readlines()
    json_dev_data = load_json(dev_content)
    clean_dev_data = lr_dev_clean_data(json_dev_data)
    to_predict = []
    answers = []
    doc_ids = []
    num = 0
    for obj in clean_dev_data:
        doc_ids.append("text{:04}".format(num))
        to_predict.append(obj['text'])
        answers.append(obj['lang'])
        num += 1
    # predicted_probabilities = pipeline.predict_proba(to_predict)
    predictions = pipeline.predict(to_predict)
    score = pipeline.decision_function(to_predict)
    final_predictions = []
    for i in range(len(predictions)):
        if max(score[i]) < -1.3:
            # print(max(score[i]))
            # print(str(predictions[i]) + " " + str(answers[i]))
            final_predictions.append('unk')
        else:
            final_predictions.append(predictions[i])

    count = 0
    total = 0
    unks_wrong = 0
    max_score_unk = -1000;
    resultFile = open("lr-dev.csv", "w")
    resultFile.write("docid,lang\n")
    for i in range(len(final_predictions)):
        resultFile.write(doc_ids[i] + "," + final_predictions[i] + "\n")

    for i in range(len(answers)):
        if final_predictions[i] == answers[i]:
            count += 1
        else:
            # print("Predicted:" + str(final_predictions[i]) + " Real:" + str(answers[i]))
            # print(answers[i] + " " + predictions[i])
            # print(str(predicted_probabilities[i]))
            if answers[i] == 'unk':
                if max(score[i]) > max_score_unk:
                    max_score_unk = max(score[i])
                unks_wrong += 1
        total += 1
    print("Correct: " + str(count) + " Total: " + str(total) + " Unks wrong: "  + str(unks_wrong))
    print("Working on dev data done, moving on to test data")
    with open("project2/test.json") as file:
        dev_content = file.readlines()
    json_dev_data = load_json(dev_content)
    clean_dev_data = lr_dev_clean_data(json_dev_data)
    to_predict = []
    answers = []
    doc_ids = []
    for obj in clean_dev_data:
        doc_ids.append(obj['id'])
        to_predict.append(obj['text'])
        # answers.append(obj['lang'])
    # predicted_probabilities = pipeline.predict_proba(to_predict)
    predictions = pipeline.predict(to_predict)
    score = pipeline.decision_function(to_predict)
    final_predictions = []
    for i in range(len(predictions)):
        if max(score[i]) < -1.0:
            # print(max(score[i]))
            # print(str(predictions[i]) + " " + str(answers[i]))
            final_predictions.append('unk')
        else:
            final_predictions.append(predictions[i])
    resultFile = open("lr-test.csv", "w")
    resultFile.write("docid,lang\n")
    for i in range(len(final_predictions)):
        resultFile.write(doc_ids[i] + "," + final_predictions[i] + "\n")
        

    
#############################################################################################
                 # Support Vector Machine and Logistic Regression #
    
def main():
    # Train and test on all three systems.
    # assumed dev.json location is the same directory as this file
    # assumed test.json is inside a directory called project2 which
    # is located in the current directory. 
    # plt.ticklabel_format(useOffset=False)
    naive_bayes_system()
    svm_system()
    lr_system()
main()


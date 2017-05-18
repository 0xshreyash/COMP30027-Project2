import json
from collections import defaultdict
from pandas import DataFrame
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
    # create_ngrams(json_data, n)
    data_frame = build_dataframe(json_data)
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])
    pipeline.fit(data_frame['text'].values, data_frame['class'].values)
    return pipeline

'''
def vectorize_counts(data_frame):
    count_vectorizer = CountVectorizer()
    counts = count_vectorizer.fit_transform(data_frame['text'].values)
    return counts
'''

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
        if final_predictions[i] == answers[i]:
            count += 1
        else:
            print("Predicted:" + str(final_predictions[i]) + " Real:" + str(answers[i]))
            print(str(max(predicted_probabilities[i])))
            if(answers[i] == 'unk'):
                unks_wrong += 1
        if(answers[i] == 'unk'):
            print(str(max(predicted_probabilities[i])))
        total += 1
    
    print(str(count) + " " + str(total) + " "  + str(unks_wrong))
    
main()



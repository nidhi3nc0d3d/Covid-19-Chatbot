import json
import nltk
import numpy
import random
import tensorflow
import tflearn
import pickle
import textwrap
import cases

from nltk.stem.lancaster import LancasterStemmer
from Screenshot import press_ss_for_screenshot

nltk.download('punkt')
stemmer = LancasterStemmer()

with open('intents.json') as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=2000, batch_size=8, show_metric=True)
model.save("model.tflearn")

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=2000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def collect_case(inp):
    collect = inp.split()
    for item in collect:
        if item.lower() == "death" or item.lower() == "deaths":
            cases.death_cases()
        if item.lower() == "total":
            cases.total()
        if item.lower() == "recover" or item.lower() == "recovered" or item.lower() == "recovery":
            cases.recover()
        if item.lower() == "active":
            cases.active()
        if item.lower() == "positive":
            for item2 in collect:
                if item2.lower() == "cases":
                    cases.active()
            break
        elif item.lower() == "cases" or item.lower() == "case":
            for i in collect:
                match = i.lower()
                if match == "covid" or match == "covid-19" or match == "coronavirus" or match == "positive":
                    if match != "mean" or match == "covid-":
                        ask = input("""
 To know all cases                   : type - yes
 To know total cases of covid-19     : type - total
 To know active cases of covid-19    : type - active
 To know recovered cases of covid-19 : type - recovered
 To know deaths during covid-19      : type - deaths
 To return to the Main Chat          : type - chat

 ANSWER ->  """)
                        if ask.lower() == "yes":
                            cases.all_cases()
                            break
                        if ask.lower() == "total":
                            cases.total()
                            break
                        if ask.lower() == "recovered":
                            cases.recover()
                            break
                        if ask.lower() == "active":
                            cases.active()
                            break
                        if ask.lower() == "deaths":
                            cases.death_cases()
                            break
                        if ask.lower() == "chat":
                            break
                        else:
                            print("Sorry, I didn't get you!")
                            break
                    else:
                        pass
            else:
                pass


Bot_name = "Bot"


def chat():
    print("WELCOME TO COVID-19 ENQUIRY")
    print("Instructions:\n(Type 'quit' to exit)!\n(Type 'ss' or 'screenshot' to take a screenshot)")
    count = 1
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        if inp.lower() == "ss" or inp.lower() == "screenshot":
            press_ss_for_screenshot(count)
            count += 1
        if inp.lower() == "cases":
            cases.all_cases()

        collect_case(inp)

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            ans = random.choice(responses)
            wrapper = textwrap.TextWrapper(width=70, break_long_words=False, replace_whitespace=False)
            ordered_text = textwrap.dedent(text=ans)
            original = wrapper.fill(text=ordered_text)
            print(f"{Bot_name}: {original}")
        else:
            print(f"{Bot_name}: I'm not sure about that. Try again.")


chat()

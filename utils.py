import json
import numpy as np
import re
from textblob import TextBlob



def data_prep(feature_num, new_feature, no_text):
    # To read and process the dataset. Return training, validation and test set as required
    with open("proj1_data.json") as fp:
        data = json.load(fp)

    # Now the data is loaded.
    # 12000 data points

    distone = {}

    feature_words = []
    rude_words = ['shit','fuck','fucking','stupid','dumb','fvck','bitch','bastard','jerk','damn','fvcking','whore']
    popular_words = ['best', 'us', 'friend', 'family', 'told', 'wife', 'said', 'someone', 'better', 'guy', 'man', 'those', 'should', 'down', 'lot', 'its', 'movie']
    # popular words are high frequency words extracted from the most popular train data
    train_num = 10000
    validate_num = 11000
    test_num = 12000

    for index in range(0, train_num):
        texts = str(data[index]['text'])  # Extract and index each text contained in the dataset
        texts = texts.lower()  # Lowercase the text
        texts = re.sub('\W', " ", texts)  # Substitute all char other than alphabet and number by space
        textstr = texts.split()  # Split texts by space
        for keyone in textstr:
            if not distone.get(keyone):
                distone[keyone] = 1
            else:
                distone[keyone] += 1  # Record the frequency of each word in a dictionary

    # Rank the frequency of each word, and make the most popular list in word+frequence pairs
    distone = sorted(distone.items(), key=lambda d: d[1], reverse=True)

    for i in distone:
        if len(feature_words) < feature_num:
            feature_words.append(i[0])  # Take the most popular featurenum words
    print(feature_words)

    full_x = np.zeros((len(data), feature_num + 3))
    full_y = np.zeros(len(data))  # Initiate blank full feature set and true value

    new_feature1 = np.zeros(len(data))  # rude words
    new_feature2 = np.zeros(len(data))  # comments length
    new_feature3 = np.zeros(len(data))  # Sentiment parameter
    new_feature4 = np.zeros(len(data))  # popular words in popular comments
    # construct 4 new features if needed

    for index in range(0, len(data)):

        full_y[index] = data[index]['popularity_score']  # True value is the popularity score
        texts = str(data[index]['text'])
        texts = texts.lower()
        texts = re.sub('\W', " ", texts)
        if new_feature == 'Y':
            textsblob=TextBlob(texts)
            new_feature3[index]=textsblob.sentiment[1]*5.0
        textstr = texts.split()  # Lower case and split each text content again
        new_feature2[index]=len(textstr)/50.0
        for keyone in textstr:
            for j in range(0, feature_num):
                if keyone == feature_words[j]:
                    full_x[index, j] += 1  # For each occurence of 160 popular words in the text, add 1
            for k in range(0,len(rude_words)):
                if keyone == rude_words[k]:
                    new_feature1[index] += 1
            for x in range(0,len(popular_words)):
                if keyone == popular_words[x]:
                    new_feature4[index] += 1

        full_x[index, feature_num] = data[index]['children']
        full_x[index, feature_num + 1] = data[index]['controversiality']
        full_x[index, feature_num + 2] = int(data[index]['is_root'])#add simple features


    if new_feature == 'Y':
        new_feature1 = new_feature1.reshape(len(data), 1)
        new_feature2 = new_feature2.reshape(len(data), 1)
        new_feature3 = new_feature3.reshape(len(data), 1)
        new_feature4 = new_feature4.reshape(len(data), 1)
        full_x = np.hstack((new_feature1, full_x))
        full_x = np.hstack((new_feature2, full_x))
        full_x = np.hstack((new_feature3, full_x))
        full_x = np.hstack((new_feature4, full_x))


    if no_text == 'Y':
        full_x = np.hstack((np.ones((len(data), 1), dtype=int), full_x[:, -4:-1]))  # Add bias term as the first column
    elif no_text == 'N':
        full_x = np.hstack((np.ones((len(data), 1), dtype=int), full_x))  # Add bias term as the first column


    full_y = full_y.reshape(len(data), 1)  # Reshape Y into column vector

    train_x, train_y = full_x[0:train_num, :], full_y[0:train_num, :]  # Define training set
    valid_x, valid_y = full_x[train_num:validate_num, :], full_y[train_num:validate_num, :]  # Define validation set
    test_x, test_y = full_x[validate_num:test_num, :], full_y[validate_num:test_num, :]  # Define test set
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def cf_train(train_x, train_y):
    #Closed_form solution
    w = np.linalg.inv(train_x.T @ train_x) @ train_x.T @ train_y
    return w


def calc_mse(w, validate_x, validate_y):
    return np.mean(np.square(validate_x @ w - validate_y))

def contrast_mse(validate_y):
    return np.mean(np.square(validate_y))

def gd_train(train_x, train_y, validate_x, validate_y, theta, beta, criteria, w0):
    # Gradient_descent
    i = 1
    prev_w = w0
    hist_mse = []

    while True:
        a = theta / (1 + beta * i);  # Define or update learning rate

        #prev_err = (train_y - train_x @ prev_w).T @ (train_y - train_x @ prev_w)  # Compute previous error
        w = prev_w - 2 * a * (train_x.T @ train_x @ prev_w - train_x.T @ train_y)  # Update weights
        #err = (train_y - train_x @ w).T @ (train_y - train_x @ w)  # Compute new error
        gd_mse = calc_mse(w, train_x, train_y)  # Computing performance using MSE
        print("gd_mse:", gd_mse)
        hist_mse.append(gd_mse)
        if (np.linalg.norm(w-prev_w) <= criteria) or (i >= 3000):
            break  # Break from loop once criteria is satified

        i += 1  # Update count
        prev_w = w

    print("GD times:", i)
    return w, hist_mse

# this function is just used for extracting feature words in most
# popular comments of training set(this word list has no common with words feature.)
def new_feature2(data, train_num, feature_num, feature_words):
    train_data = data[0:train_num];

    sorted_data = sorted(train_data, key=lambda x: x['popularity_score'], reverse=True)
    top_ones = sorted_data[0:2000]
    pop_feature_words = text_extract(top_ones, feature_num)
    diff = [i for i in pop_feature_words if
            i not in feature_words]  # Find words that are not included in the most popular words already
    # print(diff)

    new_feature = np.zeros((len(data), len(diff)))

    for index in range(0, len(data)):
        texts = str(data[index]['text'])
        texts = texts.lower()
        texts = re.sub('\W', " ", texts)
        textstr = texts.split()  # Lower case and split each text content again
        for keyone in textstr:
            for j in range(0, len(diff)):
                if keyone == diff[j]:
                    new_feature[index, j] += 1  # For each occurence of top popular words in the text, add 1

    return new_feature  # In frequency matrix


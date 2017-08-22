import numpy as np
from math import *
import random
import operator

def load_dataSet():
    posting_list = np.array([
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ])
    class_vector = [0,1,0,1,0,1]
    return posting_list,class_vector

def create_vocab_list(dataSet):
    vocab_set = set([])
    for data in dataSet:
        vocab_set = vocab_set | set(data)
    return list(vocab_set)

def words_to_vocab_vector(input_Set,vocab_list):
    return_vector = np.zeros(len(vocab_list))
    for word in input_Set:
        if word in vocab_list:
            return_vector[vocab_list.index(word)] += 1
        # else:
        #     print 'the word: %s is not in the vocabulary!' %word
    return return_vector

def train_NB(train_matrix, train_class_list):
    unique_class_list = list(set(train_class_list))
    pClass = np.zeros(len(unique_class_list))
    for item in unique_class_list:
        pClass[unique_class_list.index(item)] = train_class_list.count(item) / float(len(train_class_list))
    num_words = len(train_matrix[0])
    vocab_in_class_array = np.ones((len(unique_class_list),num_words))
    sum_words_in_class = np.ones(len(unique_class_list))
    for item in unique_class_list:
        item_index = unique_class_list.index(item)
        for i in range(len(train_class_list)):
            if train_class_list[i] == item:
                vocab_in_class_array[item_index] += train_matrix[i]
                sum_words_in_class[item_index] += sum(train_matrix[i])
    p_vocab_in_class_array = []
    for i in range(len(unique_class_list)):
        p_vocab_in_class_array.append(vocab_in_class_array[i] / float(sum_words_in_class[i]))
    p_vocab_in_class_array = np.log(p_vocab_in_class_array)
    p_vocab_in_class_array = np.array(p_vocab_in_class_array)

    return p_vocab_in_class_array,pClass,unique_class_list

def classify_NB(test_vector,p_vocab_in_class_array,pClass,unique_class_list):
    p = np.zeros(len(unique_class_list))
    for i in range(len(unique_class_list)):
         p[i] = sum(test_vector * p_vocab_in_class_array[i]) + log(pClass[i])
    return unique_class_list[np.argmax(p)]

def test_NB():
    posting_dataSet,class_list = load_dataSet()
    my_vocab = create_vocab_list(posting_dataSet)
    train_matrix = []
    for data in posting_dataSet:
        train_matrix.append(words_to_vocab_vector(data,my_vocab))
    train_matrix = np.array(train_matrix)
    p_vocab_in_class_array,pClass,unique_class_list = train_NB(train_matrix,class_list)

    test_words = np.array([['love', 'my', 'dalmation'],
                           ['stupid', 'garbage'],
                           ['help','cute','stupid'],
                           ['help', 'cute', 'garbage']])
    test_array = []
    for data in test_words:
        test_array.append(words_to_vocab_vector(data,my_vocab))
    test_array = np.array(test_array)

    classified_result = []
    for i in range(test_array.shape[0]):
        classified_result.append(classify_NB(test_array[i],p_vocab_in_class_array,pClass,unique_class_list))
        print test_words[i], 'classified as:', classified_result[i]

def text_Parse(text):
    import re
    words_list = re.split(r'\W+',text)
    return [word.lower() for word in words_list if len(word) > 2]

def test_spam():
    doc_list = []
    class_list = []
    for i in range(1,26):
        parsed_text = text_Parse(open('email/ham/%d.txt' %i).read())
        doc_list.append(parsed_text)
        class_list.append(0)
        parsed_text = text_Parse(open('email/spam/%d.txt' %i).read())
        doc_list.append(parsed_text)
        class_list.append(1)

    vocab = create_vocab_list(doc_list)

    error_count_list = [0] * 10
    for j in range(10):
        training_index = range(50)
        test_index = [0] * 10
        for i in range(10):
            index = int(random.uniform(0,len(training_index)))
            test_index[i] = training_index[index]
            del(training_index[index])

        train_matrix = np.zeros((len(training_index),len(vocab)))
        train_class_list = [0] * len(training_index)
        for i in range(len(training_index)):
            index = training_index[i]
            train_matrix[i] = words_to_vocab_vector(doc_list[index],vocab)
            train_class_list[i] = class_list[index]

        p_vocab_in_class_array, pClass, unique_class_list = train_NB(train_matrix, train_class_list)

        error_count = 0
        for i in range(len(test_index)):
            index = test_index[i]
            test_veotor = words_to_vocab_vector(doc_list[index],vocab)
            classified_result = classify_NB(test_veotor,p_vocab_in_class_array,pClass,unique_class_list)
            if classified_result != class_list[index]:
                error_count += 1
                print "classification error", doc_list[index]
        error_count_list[j] = float(error_count) / len(test_index)
        print 'the error rate is:', error_count_list[j]

    print 'the mean error rate is:%.3f' %np.mean(error_count_list)

def test_local_words(feed0,feed1):
    import feedparser
    min_num = min(len(feed0['entries']),len(feed1['entries']))
    doc_list = []
    class_list = []
    fulltext = []
    for i in range(min_num):
        parsed_text = text_Parse(feed0['entries'][i]['summary'])
        doc_list.append(parsed_text)
        fulltext.extend(parsed_text)
        class_list.append(0)
        parsed_text = text_Parse(feed1['entries'][i]['summary'])
        doc_list.append(parsed_text)
        fulltext.extend(parsed_text)
        class_list.append(1)

    vocab = create_vocab_list(doc_list)
    top30_words = calc_most_freq(vocab,fulltext)
    for (word,count) in top30_words:
        if word in vocab:
            vocab.remove(word)

    # error_count_list = [0] * 10
    # for j in range(10):
    num_test = int(0.1 * len(doc_list))
    training_index = range(len(doc_list))
    testing_index = [0] * num_test
    for i in range(num_test):
        index = int(random.uniform(0,len(training_index)))
        testing_index[i] = training_index[index]
        del(training_index[index])

    train_matrix = np.zeros((len(training_index),len(vocab)))
    train_class_list = [0] * len(training_index)
    for i in range(len(training_index)):
        index = training_index[i]
        train_matrix[i] = words_to_vocab_vector(doc_list[index],vocab)
        train_class_list[i] = class_list[index]
    p_vocab_in_class_array,pClass,unique_class_list = train_NB(train_matrix,train_class_list)

    # error_count = 0.0
    # for i in range(len(testing_index)):
    #     index = testing_index[i]
    #     test_veotor = words_to_vocab_vector(doc_list[index],vocab)
    #     classified_result = classify_NB(test_veotor,p_vocab_in_class_array,pClass,unique_class_list)
    #     if classified_result != class_list[index]:
    #         error_count += 1
    #         print 'classification error', doc_list[index]
    #
    # error_count_list[j] = float(error_count) / len(testing_index)
    # print 'the error rate is:', error_count_list[j]

    # print 'the mean error rate is:%.3f' %np.mean(error_count_list)
    return vocab,p_vocab_in_class_array


def calc_most_freq(vocab,fulltext):
    vocab_count = {}
    for word in fulltext:
        vocab_count[word] = fulltext.count(word)
    sorted_vocab_count = sorted(vocab_count.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sorted_vocab_count[0:10]

def get_top_words(feed0,feed1):
    vocab,p_in_class_array = test_local_words(feed0,feed1)
    top_words = []
    for i in range(p_in_class_array.shape[0]):
        for j in range(p_in_class_array.shape[1]):
            if p_in_class_array[i,j] > -5.0:
                top_words.append((vocab[j],p_in_class_array[i,j]))
        sorted_top_words = sorted(top_words, key=lambda a:a[1], reverse=True)
        print '--------------------------------------------------------'
        for item in sorted_top_words:
            print item[0]


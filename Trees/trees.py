from math import log
import numpy as np
import operator

def cal_shannonEnt(dataSet):
    labels_count = {}
    for data in dataSet:
        label = data[-1]
        labels_count[label] = labels_count.get(label,0) + 1

    shannonEnt = 0.0
    num_dataset = len(dataSet)
    for key,value in labels_count.items():
        prob = float(value) / num_dataset
        shannonEnt -= prob * log(prob,2)

    return shannonEnt

def create_dataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

def split_Dataset(dataSet,index,value):
    reduced_dataSet = []
    for data in dataSet:
        if data[index] == value:
            reduced_vector = data[:index]
            reduced_vector.extend(data[index+1:])
            reduced_dataSet.append(reduced_vector)

    return reduced_dataSet

def choose_best_feature(dataSet):
    num_features = len(dataSet[0]) - 1
    base_ent = cal_shannonEnt(dataSet)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        unique_vals = [data[i] for data in dataSet]
        unique_vals = set(unique_vals)
        new_ent = 0.0
        for value in unique_vals:
            sub_dataSet  = split_Dataset(dataSet,i,value)
            prob = float(len(sub_dataSet)) / len(dataSet)
            new_ent += prob * cal_shannonEnt(sub_dataSet)

        info_gain = base_ent - new_ent
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature

def majority_voting(class_list):
    class_count = {}
    for vote in class_list:
        class_count[vote] = class_count.get(vote,0) + 1
    sorted_class_count = sorted(class_count.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sorted_class_count[0][0]

def create_Trees(dataSet,labels):
    feature_labels = labels[:]
    class_list = [data[-1] for data in dataSet]
    if class_list.count(class_list[-1]) == len(class_list):
        return class_list[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(class_list)

    best_feature_index = choose_best_feature(dataSet)
    best_feature_label = feature_labels[best_feature_index]
    del(feature_labels[best_feature_index])
    mytree = {best_feature_label:{}}
    # why is wrong?
    # unique_vals = set()
    # unique_vals.clear()
    # unique_vals.add(data[best_feature_index] for data in dataSet)
    feature_vector = [data[best_feature_index] for data in dataSet]
    unique_vals = set(feature_vector)
    for value in unique_vals:
        sub_labels = feature_labels[:]
        mytree[best_feature_label][value] = create_Trees(split_Dataset(dataSet,best_feature_index,value),sub_labels)

    return mytree

def classify_using_trees(input_trees,feature_labels,test_vector):
    feature = input_trees.keys()[0]
    feature_index = feature_labels.index(feature)
    second_dict = input_trees[feature]
    for key in second_dict.keys():
        if test_vector[feature_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify_using_trees(second_dict[key],feature_labels,test_vector)
            else:
                class_label = second_dict[key]

    return class_label

def classify_using_trees2(input_trees,feature_labels,test_vector):
    feature = input_trees.keys()[0]
    feature_index = feature_labels.index(feature)
    key = test_vector[feature_index]
    second_dict = input_trees[feature]
    if isinstance(second_dict[key],dict):
        class_label = classify_using_trees(second_dict[key],feature_labels,test_vector)
    else:
        class_label = second_dict[key]

    return class_label

def store_trees(input_trees,filename):
    import pickle
    fh = open(filename,'w')
    pickle.dump(input_trees,fh)
    fh.close()

def load_trees(filename):
    import pickle
    fh = open(filename)
    return pickle.load(fh)
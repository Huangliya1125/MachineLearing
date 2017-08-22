import numpy as np
import operator
from os import listdir

def DataSet():
    group = np.array([[1.0,1.1],
                      [1.0,1.0],
                      [0,0,],
                      [0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataset,labels,k):
    datasetsize = dataset.shape[0]
    diffmat = np.tile(inX,(datasetsize,1)) - dataset
    dist = ((diffmat**2).sum(axis=1))**0.5
    sorted_dist_index = dist.argsort()
    classcount = {}
    for i in range(k):
        label = labels[sorted_dist_index[i]]
        classcount[label] = classcount.get(label,0) + 1

    sorted_classcount = sorted(classcount.iteritems(),key = operator.itemgetter(1),reverse=True)
    return sorted_classcount[0][0]

def file_to_matrix(filename):
    fh = open(filename)
    num_lines = len(fh.readlines())
    returnMat = np.zeros((num_lines,3))
    classlabels = []
    # repeat (why?)
    fh = open(filename)
    for i in range(num_lines):
        line = fh.readline()
        line = line.strip()
        list_line = line.split('\t')
        returnMat[i:] = list_line[:3]
        classlabels.append(int(list_line[-1]))

    return returnMat,classlabels

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    rangeVals = maxVals - minVals
    normed_dataSet = np.zeros((dataSet.shape))
    num_row = dataSet.shape[0]
    normed_dataSet = (dataSet - np.tile(minVals,(num_row,1))) / (np.tile(rangeVals,(num_row,1)))
    return normed_dataSet,rangeVals,minVals

def dating_Class_Test():
    test_ratio = 0.1
    dating_matrix,dating_labels = file2matrix('datingTestSet2.txt')
    num_row = dating_matrix.shape[0]
    num_test = int(test_ratio * num_row)
    classified_results = []
    error_count = 0.0
    for i in range(num_test):
        classified_results.append(classify0(dating_matrix[i,:],dating_matrix[num_test:num_row,:],dating_labels[num_test:num_row],3))
        print 'the classifier gave the result: %d , while the real result is: %d' %(classified_results[i],dating_labels[i])
        if classified_results[i] != dating_labels[i]:
            error_count += 1.0
    print 'the total error rate is: %f' % (error_count / float(num_row))

def classify_person():
    results_list = ['not at all', 'in small doses', 'in large doses']
    precent_playing = float(raw_input('percent of time spent playing video games:'))
    ffmiles = float(raw_input('frequemt flier miles earned per year:'))
    icecream = float(raw_input('liters of ice cream consumed per year:'))
    inX = np.array([precent_playing, ffmiles, icecream])
    dating_matrix, dating_labels = file2matrix('datingTestSet2.txt')
    normed_dating_matrix, ranges, minvals = autoNorm(dating_matrix)
    classified_result = classify0((inX - minvals) / ranges, normed_dating_matrix, dating_labels, 3)
    print 'You will probably like this person:', results_list[classified_result - 1]

def img_to_vector(filename):
    return_vector = np.zeros((1,1024))
    fh = open(filename)
    num_lines = len(fh.readlines())
    fh = open(filename)
    for i in range(num_lines):
        line = fh.readline()
        for j in range(num_lines):
            return_vector[0,i*num_lines+j] = int(line[j])

    return return_vector

def handwritting_class_Test():
    training_files_list = listdir('trainingDigits')
    num_training_files = len(training_files_list)
    training_matrix = np.zeros((num_training_files,1024))
    training_labels = []
    for i in range(num_training_files):
        training_matrix[i,:] = img_to_vector('trainingDigits/%s' %training_files_list[i])
        label = int((training_files_list[i].split('.')[0]).split('_')[0])
        training_labels.append(label)

    test_files_list = listdir('testDigits')
    num_test_files = len(test_files_list)
    classified_results = []
    error_count = 0.0
    for i in range(num_test_files):
        inX = img_to_vector('testDigits/%s' %test_files_list[i])
        classified_results.append(classify0(inX,training_matrix,training_labels,3))
        label = int((test_files_list[i].split('.')[0]).split('_')[0])
        print 'the classifier gave the result: %d, while the real result is: %d' %(classified_results[i],label)
        if classified_results[i] != label:
            error_count += 1.0

    print 'the total number of errors is: %d' %error_count
    print 'the total error rate is: %f' %(error_count/float(num_test_files))
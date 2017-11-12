import sys
from ID3 import buildTreeID3
import Node
from random import randrange
from random import randint
import numpy as np
from ann import ann_model, predict
import ann
from nbayes import getBinValues, getFrequencyDict, getProbability, getPredictions
from logreg import sigmoid, initializeParameters, propagate, optimize, predict, model


# Returns the different values of a particular feature
def getValues(data, features, feature):
    i = features.index(feature)
    diff_values = {}
    change_to = 1
    for each_row in data:
        if each_row[i] not in diff_values.keys():
            diff_values[each_row[i]] = change_to
            change_to += 1
    return diff_values

def changeDataToContinuousFeatures(data, features, feature, diff_values):
    changed_data = []
    i = features.index(feature)
    for each_row in data:
        each_row[i] = int(diff_values[each_row[i]])
        changed_data.append(each_row)
    return changed_data

#Returns the data as a list of lists after reading the data file
def getData(data_file, data_type, remove_index):
    data = [[]]
    count = 0
    for each_row in data_file:
        count += 1
        each_row = each_row.strip("\r\n")
        list_of_each_row = each_row.split(',')
        list_of_each_row[-1] = int(list_of_each_row[-1])
        list_of_each_row = list_of_each_row[remove_index:]
        for each_value in list_of_each_row:
            if data_type[list_of_each_row.index(each_value)] == 'continuous':
                list_of_each_row[list_of_each_row.index(each_value)] = round(float(list_of_each_row[list_of_each_row.index(each_value)]),2)
        data.append(list_of_each_row)
    data.remove([])
    return data

#This function takes in examples and decision_tree and returns the predicted_values
def classifyExamples(examples, features, decision_tree):
    temp_dict = decision_tree.copy()
    result = ""
    predicted_values = []
    for example in examples:
        temp_dict = decision_tree.copy()
        while(isinstance(temp_dict, dict)):
            root = Node.Node(temp_dict.keys()[0], temp_dict[temp_dict.keys()[0]])
            temp_dict = temp_dict[temp_dict.keys()[0]]
            index = features.index(root.feature)
            value = example[index]
            if(value in temp_dict.keys()):
                child = Node.Node(value, temp_dict[value])
                result = temp_dict[value]
                temp_dict = temp_dict[value]
    	    else:
                result = "?"
                break
        predicted_values.append(result)
    return predicted_values

#Returns the values of the column as a list
def getColumnValues(data, index):
    column_values = []
    for each_row in data:
        #print each_row
        column_values.append(each_row[index])
    return column_values

#Returns the data after joinging all the splits
def joinFolds(data):
    joined_folds = []
    for each_set in data:
        for each_row in each_set:
            joined_folds.append(each_row)
    return joined_folds

#Returns the accuracy of the classified examples
def getAccuracy(true_values, predicted_values):
    correct = 0.0
    for i in range(0, len(true_values)):
        if true_values[i] == predicted_values[i]:
            correct += 1
    return correct/len(true_values)

#Returns the dataset which is split depending on the different class values
def splitDataOnClass(data):
    class_example_dict = {}
    for each_row in data:
        if class_example_dict.has_key(each_row[-1]):
            class_example_dict[each_row[-1]].append(each_row)
        else:
            class_example_dict[each_row[-1]] = [each_row]
    return class_example_dict

#Creates the 5 fold stratified cross validation set
def createFolds(data, data_set_split, folds):
    dataset_copy = list(data)
    fold_size = int(len(data) / folds)
    if len(data_set_split) == 0:
        for i in range(folds):
    		fold = []
    		while len(fold) < fold_size:
    			index = randrange(len(dataset_copy))
    			fold.append(dataset_copy.pop(index))
    		data_set_split.append(fold)
    else:
        final_size = len(data_set_split[0]) + fold_size
        for i in range(0, len(data_set_split)):
            while len(data_set_split[i]) < final_size:
                index = randrange(len(dataset_copy))
                data_set_split[i].append(dataset_copy.pop(index))
    return data_set_split

#Returns the dictionary with whether the feature is required for depth or not
def buildFeatureDepthIndex(features, features_on_depth):
    features_in_depth_index = []
    for feature_in_depth in features_on_depth:
        features_in_depth_index.append(features.index(feature_in_depth))
    return features_in_depth_index

def buildDataOnDepth(train_set, features_in_depth_index):
    data = [[]]
    for each_row in train_set:
        changed_row = []
        for i in range(0, len(each_row)):
            if i in features_in_depth_index:
                changed_row.append(each_row[i])
        data.append(changed_row)
    data.remove([])
    return data

def changeData(data, average_values, continuous_feature_index):
    data_copy = []
    for each_row in data:
        for i in range(0, len(continuous_feature_index)):
            index = continuous_feature_index[i]
            if each_row[index] <= average_values[i]:
                each_row[index] = '0'
            else:
                each_row[index] = '1'
        data_copy.append(each_row)
    return data_copy

def getThresholdValues(sorted_data,index):
    first_value = sorted_data[0][index]
    prev_target_value = sorted_data[0][-1]
    thresh_hold_values = []
    for each_row in sorted_data:
        if each_row[-1] != prev_target_value and each_row[index] not in thresh_hold_values:
            thresh_hold_values.append(each_row[index])
            prev_target_value = each_row[-1]
    return thresh_hold_values

def generateBootstrap(data):
    data_copy = []
    while len(data_copy) < len(data):
        i = randint(0, len(data)-1)
        data_copy.append(data[i])
    return data_copy

def convertToFloat(data, features, each_feature):
    changed_data = []
    i = features.index(each_feature)
    for each_row in data:
        each_row[i] = float(each_row[i])
        changed_data.append(each_row)
    return changed_data

'''
Helper Functions Above
Code Execution Starts from here
'''

#These are the different arguments taken from the user
#Argument 1 will be the path to the data
#Argument 2 will be either 0 or 1. 1 - cross validation, 0 - by default, full sample
#Argument 3 will set the classifier
#Argument 4 will be the number of iterations
arguments = sys.argv
data_path = arguments[1]
cross_validation = arguments[2]
classifier = arguments[3]
iters = int(arguments[4])
names_file = open(data_path + '.names')
data_file = open(data_path + '.data')
data = [[]]
features = []
data_type = []
for each_row in names_file:
    each_row = each_row.strip("\r\n")
    all_values = each_row.split(',')

for each_value in all_values:
    each_value = each_value.split(':')
    features.append(each_value[0])
    data_type.append(each_value[1])
#Need to remve the index column from the data. The volcanoes data set has 2 set of indexes.
remove_index = 1
if data_path == 'volcanoes':
    remove_index = 2
features = features[remove_index:]
target_feature = features[-1]
data_type = data_type[remove_index:]
data = getData(data_file, data_type, remove_index)

print features

if classifier == 'logreg' or classifier == 'ann':
    features = features[:-1]

if classifier == 'ann' or classifier == 'logreg':
    for each_feature in features:
        if data_type[features.index(each_feature)] == 'discrete':
            diff_values = getValues(data, features, each_feature)
            data = changeDataToContinuousFeatures(data, features, each_feature, diff_values)
        else:
            data = convertToFloat(data, features, each_feature)

if classifier == 'dtree':
    continuous_feature_index = []
    for i in range(0, len(data_type)):
        if data_type[i] == 'continuous':
            continuous_feature_index.append(i)
    all_features_thresh_hold = []
    thresh_hold_values = []
    middle_thresh_hold_values = []

    for index in continuous_feature_index:
        sorted_data = sorted(data, key=lambda row:row[index])
        thresh_hold_values = getThresholdValues(sorted_data, index)
        all_features_thresh_hold.append(thresh_hold_values)
        middle_thresh_hold_values.append(thresh_hold_values[len(thresh_hold_values)/2])

    data = changeData(data, middle_thresh_hold_values, continuous_feature_index)

if classifier == 'nbayes':
    total_one_count = 0
    total_count = 0
    total_zero_count = 0
    for each_row in data:
        if each_row[-1] == 1:
            total_one_count += 1
        else:
            total_zero_count += 1
        total_count += 1
    probability_zero = total_zero_count / float(total_count)
    probability_one = total_one_count / float(total_count)

    continuous_feature_index = []
    for i in range(0, len(data_type)):
        if data_type[i] == 'continuous':
            continuous_feature_index.append(i)

    feature_index_bin_values = {}
    sorted_data = data
    for index in continuous_feature_index:
        sorted_data = sorted(sorted_data, key=lambda row:row[index])
        bin_values = getBinValues(sorted_data, index, number_of_bins)
        sorted_data = changeData(sorted_data, index, bin_values)
        feature_index_bin_values[index] = bin_values

    data = sorted_data

data_sets = [data]

#Construct different data sets for bagging
for i in range(1, iters):
    data_sets.append(generateBootstrap(data))

accuracy = []
size_of_tree = []
first_feature_list = []
all_mean_accuracy = []

for data in data_sets:

    if cross_validation == '1' and classifier == 'dtree':
        temp_dict = splitDataOnClass(data)
        data_split_on_class = []
        for each_key in temp_dict.keys():
            data_split_on_class.append(temp_dict[each_key])

        data_set_split = []
        for each_split in data_split_on_class:
            data_set_split = createFolds(each_split, data_set_split, 5)

        for i in range(0, 5):
            best_feature_dict = {}
            train_set = data_set_split[:]
            test_set = train_set.pop(i)
            train_set = joinFolds(train_set)
            test_target_values = getColumnValues(test_set, features.index(target_feature))
            decision_tree, best_feature_dict = buildTreeID3(train_set, features, target_feature, split_criteria, best_feature_dict, 0)
            features_on_depth = []
            for each_feature in best_feature_dict.keys():
                features_on_depth.insert(best_feature_dict[each_feature], each_feature)
            max_depth_specified = int(max_depth_specified)
            if max_depth_specified > 0:
                features_on_depth = features_on_depth[:max_depth_specified]
                features_on_depth.append(target_feature)
                features_in_depth_index = buildFeatureDepthIndex(features, features_on_depth)
                train_set_on_depth = buildDataOnDepth(train_set, features_in_depth_index)
                decision_tree, best_feature_dict = buildTreeID3(train_set_on_depth, features_on_depth, target_feature, split_criteria, best_feature_dict, 0)
                first_feature_list.append(decision_tree.keys()[0])
                test_set_on_depth = buildDataOnDepth(test_set, features_in_depth_index)
                result_list = classifyExamples(test_set_on_depth, features_on_depth, decision_tree)
                accuracy.append(getAccuracy(test_target_values, result_list))
            else:
                size_of_tree.append(len(best_feature_dict.keys()))
                first_feature_list.append(decision_tree.keys()[0])
                result_list = classifyExamples(test_set, features, decision_tree)
                accuracy.append(getAccuracy(test_target_values, result_list))
        all_mean_accuracy.append(sum(accuracy)/len(accuracy))

    elif cross_validation == '0' and classifier == 'dtree':
        best_feature_dict = {}
        test_target_values = getColumnValues(data, features.index(target_feature))
        decision_tree, best_feature_dict = buildTreeID3(data, features, target_feature, split_criteria, best_feature_dict, 0)
        features_on_depth = []
        for each_feature in best_feature_dict.keys():
            features_on_depth.insert(best_feature_dict[each_feature], each_feature)

        max_depth_specified = int(max_depth_specified)
        if max_depth_specified > 0:
            features_on_depth = features_on_depth[:max_depth_specified]
            features_on_depth.append(target_feature)
            features_in_depth_index = buildFeatureDepthIndex(features, features_on_depth)
            data_set_on_depth = buildDataOnDepth(data, features_in_depth_index)
            decision_tree, best_feature_dict = buildTreeID3(data_set_on_depth, features_on_depth, target_feature, split_criteria, best_feature_dict, 0)
            first_feature_list.append(decision_tree.keys()[0])
            result_list = classifyExamples(data_set_on_depth, features_on_depth, decision_tree)
            accuracy.append(getAccuracy(test_target_values, result_list))
        else:
            size_of_tree.append(len(best_feature_dict.keys()))
            first_feature_list.append(decision_tree.keys()[0])
            result_list = classifyExamples(data, features, decision_tree)
            accuracy.append(getAccuracy(test_target_values, result_list))
        all_mean_accuracy.append(mean(accuracy))

    elif cross_validation == '1' and classifier == 'ann':
            temp_dict = splitDataOnClass(data)
            data_split_on_class = []
            for each_key in temp_dict.keys():
                data_split_on_class.append(temp_dict[each_key])

            data_set_split = []
            for each_split in data_split_on_class:
                data_set_split = createFolds(each_split, data_set_split, 5)

            for i in range(0, len(data_set_split)):
                train_set = data_set_split[:]

                test_set = np.array(train_set.pop(i))
                train_set = np.array(joinFolds(train_set))
                train_set = train_set.transpose()
                train_labels = np.array([train_set[-1,:]])

                test_set = test_set.transpose()
                test_labels = np.array([test_set[-1,:]])
                train_set = train_set[1:-1,:]
                test_set = test_set[1:-1,:]

                parameters = ann.ann_model(train_set, train_labels, size_of_hidden_layer = 2, num_of_iterations=1000, weight_decay=0.1)
                predictions = ann.predict(parameters, test_set)
                converted_predictions = []
                for each_prediction in predictions[0]:
                    if each_prediction == False:
                        converted_predictions.append(0)
                    else:
                        converted_predictions.append(1)
                accuracy.append(getAccuracy(test_labels[0], converted_predictions))
            all_mean_accuracy.append(sum(accuracy)/len(accuracy))

    elif cross_validation == '0' and classifier == 'ann' :
        parameters = ann.ann_model(input_data, labels, size_of_hidden_layer = 2, num_of_iterations=1000, weight_decay=0.1)
        predictions = ann.predict(parameters, input_data)
        print predictions
        converted_predictions = []
        for each_prediction in predictions[0]:
            if each_prediction == False:
                converted_predictions.append(0)
            else:
                converted_predictions.append(1)
        accuracy.append(getAccuracy(labels[0], converted_predictions))
        all_mean_accuracy.append(sum(accuracy)/len(accuracy))

    elif cross_validation == '1' and classifier == 'nbayes':

        temp_dict = splitDataOnClass(data)
        data_split_on_class = []
        for each_key in temp_dict.keys():
            data_split_on_class.append(temp_dict[each_key])

        data_set_split = []
        for each_split in data_split_on_class:
            data_set_split = createFolds(each_split, data_set_split, 5)

        accuracy_values = []
        for i in range(0, len(data_set_split)):
            train_set = data_set_split[:]
            test_set = train_set.pop(i)
            train_set = joinFolds(train_set)

            frequencies_dict = {}
            probability_dict_each_Y = {}
            probability_dict_each_value = {}
            for each_feature in features:
                values = getValues(train_set, features, each_feature)
                frequencies_dict[each_feature] = getFrequencyDict(data, features, each_feature, values)
                probability_dict_each_Y[each_feature], probability_dict_each_value[each_feature] = getProbability(frequencies_dict[each_feature], total_count)

            true_values = getColumnValues(test_set, -1)
            predictions = getPredictions(test_set, features, probability_dict_each_Y, probability_dict_each_value, probability_zero, probability_one)
            accuracy = getAccuracy(true_values, predictions)
            accuracy_values.append(accuracy)
        all_mean_accuracy.append(sum(accuracy_values)/len(accuracy_values))

    elif cross_validation == '0' and classifier == 'nbayes':
        frequencies_dict = {}
        probability_dict_each_Y = {}
        probability_dict_each_value = {}
        for each_feature in features:
            values = getValues(data, features, each_feature)
            frequencies_dict[each_feature] = getFrequencyDict(data, features, each_feature, values)
            probability_dict_each_Y[each_feature], probability_dict_each_value[each_feature] = getProbability(frequencies_dict[each_feature], total_count)

        true_values = getColumnValues(data, -1)
        predictions = getPredictions(data, features, probability_dict_each_Y, probability_dict_each_value, probability_zero, probability_one)
        accuracy = getAccuracy(true_values, predictions)
        all_mean_accuracy.append(accuracy)

    elif cross_validation == '1' and classifier == 'logreg':
        print 'Inside Logreg'
        #print data
        temp_dict = splitDataOnClass(data)
        data_split_on_class = []
        for each_key in temp_dict.keys():
            data_split_on_class.append(temp_dict[each_key])

        data_set_split = []
        for each_split in data_split_on_class:
            data_set_split = createFolds(each_split, data_set_split, 5)

        for i in range(0, len(data_set_split)):
            train_set = data_set_split[:]

            test_set = np.array(train_set.pop(i))
            train_set = np.array(joinFolds(train_set))
            train_set = train_set.transpose()
            train_labels = np.array([train_set[-1,:]])

            test_set = test_set.transpose()
            test_labels = np.array([test_set[-1,:]])
            train_set = train_set[1:-1,:]
            test_set = test_set[1:-1,:]

            parameters = model(train_set, train_labels, test_set, test_labels, num_iterations=2000, learning_rate=0.1, lam=1)
            accuracy.append(float(parameters["accuracy"]))
        all_mean_accuracy.append(sum(accuracy)/len(accuracy))

    elif cross_validation == '0' and classifier == 'logreg':
        parameters = model(input_data, labels, input_data, labels, num_iterations=2000, learning_rate=0.1, lam=1)

print sum(all_mean_accuracy)/len(all_mean_accuracy)

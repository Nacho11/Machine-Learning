import sys
import numpy as np
from random import randrange


def getValues(data, features, feature):
    i = features.index(feature)
    diff_values = {}
    change_to = 1
    for each_row in data:
        if each_row[i] not in diff_values.keys():
            diff_values[each_row[i]] = change_to
            change_to += 1
    return diff_values

def getColumnValues(data, index):
    column_values = []
    for each_row in data:
        column_values.append(each_row[index])
    return column_values

def getFrequencyDict(data, features, each_feature, values):
    frequencies_dict = {}
    i = features.index(each_feature)
    for each_value in values:
        count_zero = 0
        count_one = 0
        for each_row in data:
            if each_row[i] == each_value:
                if each_row[-1] == 0:
                    count_zero += 1
                else:
                    count_one += 1
        frequencies_dict[each_value] = {0:count_zero, 1:count_one}
    return frequencies_dict

#This method returns the probability of each value in a feature and probability for each value given Y
def getProbability(feature_freq_dict, total_count):
    probability_dict_each_Y = {}
    probability_dict_each_value = {}
    for key in feature_freq_dict.keys():
        old_dict = feature_freq_dict[key]
        each_value_count = 0
        new_dict = {}
        for each_key in old_dict.keys():
            new_dict[each_key] = old_dict[each_key]/float(total_count)
            each_value_count += old_dict[each_key]
        probability_dict_each_Y[key] = new_dict
        probability_dict_each_value[key] = each_value_count/float(total_count)
    return probability_dict_each_Y, probability_dict_each_value

def getPredictions(data, features, probability_dict_each_Y, probability_dict_each_value, probability_zero, probability_one):
    predictions = []
    #print probability_one
    for each_row in data:
        probability_one_given_feature = []
        probability_zero_given_feature = []
        for each_feature in features:
            i = features.index(each_feature)
            probability_of_feature_for_values = probability_dict_each_value[each_feature]
            probability_of_feature_given_Y = probability_dict_each_Y[each_feature]
            probability_of_value = probability_of_feature_given_Y[each_row[i]]
            probability_of_one_given_value = 0.0
            probability_of_zero_given_value = 0.0
            for key in probability_of_value.keys():
                if key == 0:
                    probability_of_zero_given_value = probability_of_value[key]
                else:
                    probability_of_one_given_value = probability_of_value[key]
            probability_one_given_feature.append((probability_of_one_given_value * probability_one) / probability_of_feature_for_values[each_row[i]])
            probability_zero_given_feature.append((probability_of_zero_given_value * probability_zero) / probability_of_feature_for_values[each_row[i]])
        product_of_prob_one = np.prod(np.array(probability_one_given_feature))
        product_of_prob_zero = np.prod(np.array(probability_zero_given_feature))
        if product_of_prob_one > product_of_prob_zero:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions

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

#Returns the data after joinging all the splits
def joinFolds(data):
    joined_folds = []
    for each_set in data:
        for each_row in each_set:
            joined_folds.append(each_row)
    return joined_folds

def changeData(data, index, bin_values):
    j = 0
    data_copy = []
    bin_value = bin_values[j]
    partitions = len(data)/(len(bin_values))
    partition = partitions
    for i in range(0, len(data)):
        each_row = data[i]
        each_row[index] = bin_value
        if i == partitions and j < len(bin_values)-1:
            j += 1
            bin_value = bin_values[j]
            partitions += partition
        data_copy.append(each_row)
    return data_copy

def getBinValues(sorted_data, index, number_of_bins):
    bin_values = []
    partitions = (len(sorted_data)/(number_of_bins-1))
    for i in range(0, len(sorted_data), partitions):
        bin_values.append(sorted_data[i][index])
    return bin_values



#These are the different arguments taken from the user
#Argument 1 will be the path to the data
#Argument 2 will be either 0 or 1. 1 - cross validation, 0 - by default, full sample
#Argument 3 number of bins
#Argument 4 value of m

arguments = sys.argv
data_path = arguments[1]
cross_validation = arguments[2]
number_of_bins = int(arguments[3])
m = float(arguments[4])

names_file = open(data_path + '.names')
data_file = open(data_path + '.data')
data = [[]]
features = []
data_type = []
for each_row in names_file:
    each_row = each_row.strip("\r\n")
    all_values = each_row.split(',')

for each_value in all_values[1:-1]:
    each_value = each_value.split(':')
    features.append(each_value[0])
    data_type.append(each_value[1])

if data_path == 'volcanoes':
    features = features[1:]
    data_type = data_type[1:]

count = 0
for each_row in data_file:
    each_row = each_row.strip("\r\n")
    list_of_each_row = each_row.split(',')
    list_of_each_row[-1] = int(list_of_each_row[-1])
    if data_path == 'volcanoes':
        data.append(list_of_each_row[2:])
    else:
        data.append(list_of_each_row[1:])
data.remove([])

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

number_of_bins = 5
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

cross_validation = '1'
data = sorted_data


if cross_validation == '1':
    print 'Inside Cross validation'

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
    print accuracy_values
else:
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
    print accuracy

print float(sum(accuracy_values)) / len(accuracy_values)

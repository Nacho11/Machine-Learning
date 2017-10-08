import numpy as np
import sys
from ann import ann_model, predict
from random import randrange


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

def normalizeRows(x):
    x_norm = np.linalg.norm(x)
    x = x / x_norm
    return x

def convertToFloat(data, features, each_feature):
    changed_data = []
    i = features.index(each_feature)
    for each_row in data:
        each_row[i] = float(each_row[i])
        changed_data.append(each_row)
    return changed_data

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

#Returns the accuracy of the classified examples
def getAccuracy(true_values, predicted_values):
    correct = 0.0
    for i in range(0, len(true_values)):
        if true_values[i] == predicted_values[i]:
            correct += 1
    return correct/len(true_values)


#These are the different arguments taken from the user
#Argument 1 will be the path to the data
#Argument 2 will be either 0 or 1. 0 - cross validation, 1 - by default, full sample
#Argument 3 will set number of hidden units
#Argument 4 will be the weight decay
#Argument 5 will be number of iterations
arguments = sys.argv
data_path = arguments[1]
cross_validation = arguments[2]
size_of_hidden_layer = int(arguments[3])
weight_decay = float(arguments[4])
num_of_iterations = int(arguments[5])

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

count = 0
for each_row in data_file:
    each_row = each_row.strip("\r\n")
    list_of_each_row = each_row.split(',')
    list_of_each_row[-1] = int(list_of_each_row[-1])
    if data_path == 'volcanoes':
        data.append(list_of_each_row[2:])
    else:
        data.append(list_of_each_row[1:])
    count += 1
    #if count == 100:
        #break
data.remove([])

print features
print data_type

for each_feature in features:
    if data_type[features.index(each_feature)] == 'discrete':
        #print each_feature
        #print data[1]
        diff_values = getValues(data, features, each_feature)
        #print diff_values
        #diff_values = {'0': 1, '+': 2, '-': 3}
        data = changeDataToContinuousFeatures(data, features, each_feature, diff_values)
    else:
        data = convertToFloat(data, features, each_feature)
#print data

#print data

np_data = np.array(data)
print np_data
np_data = np_data.transpose()
input_data = np_data[1:-1,:]
input_data = normalizeRows(input_data)

labels = np.array([np_data[-1,:]])
labels = np.array(labels)
#print input_data.shape
#print labels.shape
#input_data = input_data.astype(int)
#labels = labels.astype(int)
#print input_data[:,1].astype(float)
#print labels[:,1]
#num_of_iterations = 10
#size_of_hidden_layer = 4
#weight_decay = 0.01

cross_validation = '1'
if cross_validation == '1':
    temp_dict = splitDataOnClass(data)
    data_split_on_class = []
    for each_key in temp_dict.keys():
        data_split_on_class.append(temp_dict[each_key])

    data_set_split = []
    for each_split in data_split_on_class:
        data_set_split = createFolds(each_split, data_set_split, 5)
    #print len(data_set_split)
    #print data_set_split
    for i in range(0, len(data_set_split)):
        #np_data = np.array(data_set_split)
        #print np_data
        #np_data = np_data.transpose()
        #input_data = np_data[1:-1,:]
        #input_data = normalizeRows(input_data)

        #labels = np.array([np_data[-1,:]])
        #labels = np.array(labels)
        #print input_data.shape
        #print labels.shape
        #input_data = input_data.astype(int)
        #labels = labels.astype(int)
        #print data_set_split

        train_set = data_set_split[:]

        #data_set_split = np.array(data_set_split).transpose()
        #print test_set.shape
        test_set = np.array(train_set.pop(i))
        train_set = np.array(joinFolds(train_set))
        train_set = train_set.transpose()
        train_labels = np.array([train_set[-1,:]])

        #print train_set.shape
        test_set = test_set.transpose()
        test_labels = np.array([test_set[-1,:]])
        train_set = normalizeRows(train_set)
        test_set = normalizeRows(test_set)
        train_set = train_set[1:-1,:]
        #print train_set.shape
        test_set = test_set[1:-1,:]
        #train_labels = np.array(data_set_split[-1,:])
        #print train_labels.shape
        print train_set.shape
        print train_labels.shape

        parameters = ann_model(train_set, train_labels, size_of_hidden_layer, num_of_iterations, weight_decay)
        predictions = predict(parameters, test_set)
        print predictions[0]
        converted_predictions = []
        for each_prediction in predictions[0]:
            if each_prediction == False:
                converted_predictions.append(0)
            else:
                converted_predictions.append(1)
        print getAccuracy(test_labels[0], converted_predictions)
else:
    print input_data.shape
    print labels.shape
    parameters = ann_model(input_data, labels, size_of_hidden_layer, num_of_iterations, weight_decay)
    predictions = predict(parameters, input_data)
    print predictions

#print parameters['W1']

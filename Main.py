import sys
from ID3 import buildTreeID3
#These are the different arguments taken from the user
#Argument 1 will be the path to the data
#Argument 2 will be either 0 or 1. 0 - cross validation, 1 - by default, full sample
#Argument 3 will set maximum depth of tree, 0 - by default, full tree
#Argument 4 will be either 0 or 1. 0 - by default, information gain. 1 - gain ratio
arguments = sys.argv
data_path = arguments[1]
#cross_validation = arguments[2]
#max_depth_specified = arguments[3]
#split_to_use = arguments[4]
names_file = open(data_path + '.names')
data_file = open(data_path + '.data')
data = [[]]
features = []
for each_row in names_file:
    each_row = each_row.strip("\r\n")
    all_values = each_row.split(',')

for each_value in all_values:
    features.append(each_value)
features.remove(features[0])
target_feature = features[-1]

#format the data
for each_row in data_file:
    each_row = each_row.strip("\r\n")
    data.append(each_row.split(','))
data.remove([])

#build the decision tree
decision_tree = buildTreeID3(data, features, target_feature)
first_feature = decision_tree.keys()
print first_feature

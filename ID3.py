#Variable names are seperated by underscore
#Function names are in camel Case
import math

count = 1
#checks all the exaples if there is only one value for the target feature
def checkAll(target_list):
    first_target = target_list[0]
    for each in target_list[1:]:
        if each == first_target:
            continue
        else:
            return 'null'
    return first_target


#returns the maximum target value of all the target values.
def countMaxTarget(target_list):
    target_count_dict = {}
    for target_value in target_list:
        if target_count_dict.has_key(target_value):
            target_count_dict[target_value] += 1
        else:
            target_count_dict[target_value] = 1
    max_target = target_count_dict.keys()[0]
    for target_value in target_count_dict.keys():
        if target_count_dict[target_value] > target_count_dict[max_target]:
            max_target = target_value
    return max_target

#Returns the information gain of a given feature - feature_for_gain
def informationGain(data, features, feature_for_gain, target_feature):

    '''Entropy of Target Feature on whole dataset -
       Entropy of target feature on subset of the dataset with
       the specific feature set to its value'''

    entropy_on_data = entropy(data, features, target_feature)
    i = features.index(feature_for_gain)

    frequency_of_values = {}
    for each_row in data:
        if frequency_of_values.has_key(each_row[i]):
            frequency_of_values[each_row[i]] += 1
        else:
            frequency_of_values[each_row[i]] = 1

    #For each of the values find the entropy and sum them to get
    #the entropy of target_feature given the feature_for_gain
    sub_entropy = 0.0
    for each_value in frequency_of_values.keys():
        probability_of_value = float(frequency_of_values[each_value]) / sum(frequency_of_values.values())
        sub_data = [each_row for each_row in data if each_row[i] == each_value]
        sub_entropy += probability_of_value * entropy(sub_data, features, target_feature)

    information_gain = entropy_on_data - sub_entropy
    return information_gain


#Returns the entropy of the data
def entropy(data, features, feature_for_entropy):
    #Find the index of the feature required
    i = features.index(feature_for_entropy)

    #Calculate the frequency of the values in the feature
    frequency_of_values = {}
    for each_row in data:
        if (frequency_of_values.has_key(each_row[i])):
            frequency_of_values[each_row[i]] += 1
        else:
            frequency_of_values[each_row[i]] = 1

    #Calculate the entropy
    entropy_of_data = 0.0
    for each_frequency in frequency_of_values.values():
        frequency_by_length = float(each_frequency)/len(data)
        entropy_of_data += (frequency_by_length * -1) * math.log(frequency_by_length, 2)
    return entropy_of_data


#Returns the gain ratio
def gainRatio(data, features, feature_for_gain, target_feature):
    entropy_of_feature = float(entropy(data, features, feature_for_gain))
    if entropy_of_feature == 0:
        return 0
    gain_ratio = informationGain(data, features, feature_for_gain, target_feature) / entropy_of_feature
    return gain_ratio



#Returns the best feature depending on the maximum information gain
def bestFeature(data, features, target_feature, split_criteria):
    if len(features) == 1:
        return ;
    best_feature = features[0]
    max_gain = 0
    #thresh_hold_best = 0.0
    for feature in features[:-1]:

        if split_criteria == '0':
            gain = informationGain(data, features, feature, target_feature)
        else:
            gain = gainRatio(data, features, feature, target_feature)
        if gain > max_gain:
            max_gain = gain
            best_feature = feature
            #if data_type[features.index(best_feature)] == 'continous':
            #    thresh_hold_best = thresh_hold
    #if data_type[features.index(best_feature)] is not 'continous':
    #    thresh_hold_best = 0.0
    return best_feature

#def bestFeatureForContinuousFeature()

# Returns the different values of a particular feature
def getValues(data, features, feature):
    i = features.index(feature)
    diff_values = []
    for each_row in data:
        if each_row[i] not in diff_values:
            diff_values.append(each_row[i])
    return diff_values

#returns the examples after removing a particular value of the given feature
def removeBestFeatureExamples(data, features, feature_to_remove, value_to_remove):
    rows = [[]]
    index_of_best_feature = features.index(feature_to_remove)
    for each_row in data:
        if each_row[index_of_best_feature] == value_to_remove:
            new_row = []
            for i in range(0, len(each_row)):
                if i != index_of_best_feature:
                    new_row.append(each_row[i])
            rows.append(new_row)
    rows.remove([])
    return rows

#Builds the decision tree
def buildTreeID3(data, features, target_feature, split_criteria, best_feature_dict, count):
    target_list = getValues(data, features, target_feature)
    label = checkAll(target_list)

    #If all the examples are of one class then the tree will have only one node of that class
    if label != 'null':
        return label, best_feature_dict

    #If there are no features then the tree will contain only the value of maximum target class
    elif len(features) == 0:
        max_target = countMaxTarget(target_list)
        return max_target, best_feature_dict

    #If the only feature left is the target Variable
    elif len(features) - 1 <= 0:
        return countMaxTarget(target_list), best_feature_dict

    #Else build the decision tree by choosing the best feature
    else:
        best_feature = bestFeature(data, features, target_feature, split_criteria)
        if not best_feature_dict.has_key(best_feature):
            best_feature_dict[best_feature] = count
            count += 1
        decision_tree = {best_feature:{}}
        values_of_best_feature = getValues(data, features, best_feature)

        for value in values_of_best_feature:
            examples = removeBestFeatureExamples(data, features, best_feature, value)
            new_feature_list = features[:]
            new_feature_list.remove(best_feature)
            sub_tree, best_feature_dict = buildTreeID3(examples, new_feature_list, target_feature, split_criteria, best_feature_dict, count)
            decision_tree[best_feature][value] = sub_tree

        return decision_tree, best_feature_dict

import numpy as np

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

def getBinValues(sorted_data, index, number_of_bins):
    bin_values = []
    partitions = (len(sorted_data)/(number_of_bins-1))
    for i in range(0, len(sorted_data), partitions):
        bin_values.append(sorted_data[i][index])
    return bin_values

# %%

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log
from collections import defaultdict, Counter
import copy
import pycountry_convert as pc
import random

# from macpath import split

"""
* ECON1660
* PS4: Trees
*
* Fill in the functions that are labaled "TODO".  Once you
* have done so, uncomment (and adjust as needed) the main
* function and the call to main to print out the tree and
* the classification accuracy.
"""


"""
* TODO: Create features to be used in your regression tree.
"""

"""************************************************************************
* function: partition_loss(subsets)
* arguments:
* 		-subsets:  a list of lists of labeled data (representing groups
				   of observations formed by a split)
* return value:  loss value of a partition into the given subsets
*
* TODO: Write a function that computes the loss of a partition for
*       given subsets
************************************************************************"""
def partition_loss(subsets):
    
    num_obs = sum(len(subset) for subset in subsets)
    loss = 0
    
    for subset in subsets:
        length = len(subset)
        total_days_funded = 0
        list_days = []
        
        for obs in subset:
            days = obs[1]
            total_days_funded += days
            list_days.append(days)
            
        list_days = np.array(list_days)
        
        mean = total_days_funded / length
        squared_sum = 0
        for x in list_days:
            squared_sum += (x - mean)**2
        
        h = num_obs * squared_sum
        loss += h
            
    return loss


"""************************************************************************
* function: partition_by(inputs, attribute)
* arguments:
* 		-inputs:  a list of observations in the form of tuples
*		-attribute:  an attribute on which to split
* return value:  a list of lists, where each list represents a subset of
*				 the inputs that share a common value of the given 
*				 attribute
************************************************************************"""
def partition_by(inputs, attribute):
	groups = defaultdict(list)
	for input in inputs:
		key = input[0][attribute]	#gets the value of the specified attribute
		groups[key].append(input)	#add the input to the appropriate group
	return groups


"""************************************************************************
* function: partition_loss_by(inputs, attribute)
* arguments:
* 		-inputs:  a list of observations in the form of tuples
*		-attribute:  an attribute on which to split
* return value:  the loss value of splitting the inputs based on the
*				 given attribute
************************************************************************"""
def partition_loss_by(inputs, attribute):
	partitions = partition_by(inputs, attribute)
	return partition_loss(partitions.values())


"""************************************************************************
* function:  build_tree(inputs, num_levels, split_candidates = None)
*
* arguments:
* 		-inputs:  labeled data used to construct the tree; should be in the
*				  form of a list of tuples (a, b) where 'a' is a dictionary
*				  of features and 'b' is a label
*		-num_levels:  the goal number of levels for our output tree
*		-split_candidates:  variables that we could possibly split on.  For
*							our first level, all variables are candidates
*							(see first two lines in the function).
*			
* return value:  a tree in the form of a tuple (a, b) where 'a' is the
*				 variable to split on and 'b' is a dictionary representing
*				 the outcome class/outcome for each value of 'a'.
* 
* TODO:  Write a recursive function that builds a tree of the specified
*        number of levels based on labeled data "inputs"
************************************************************************"""
# %%

def build_tree(inputs, num_levels, candidates, num_split_candidates):
    #make sure you can split on the same attribute in different branches
    split_candidates = copy.deepcopy(candidates)
    sampled_split_candidates = None
    if len(split_candidates) <= num_split_candidates: 
        sampled_split_candidates = split_candidates
    else:
        sampled_split_candidates = random.sample(split_candidates, num_split_candidates)
    days_till_loan_lst = [row[1] for row in inputs]
    days_till_loan = np.array(days_till_loan_lst)
    avg_days_till_loan = np.round(np.mean(days_till_loan), 2)

    # if every row has the same number of days then stop splitting
    if avg_days_till_loan == days_till_loan_lst[0]:
        return avg_days_till_loan
    if num_levels == 0 or len(split_candidates) == 0:
        return avg_days_till_loan
    
    min_loss = partition_loss_by(inputs, sampled_split_candidates[0])
    best_candidate = sampled_split_candidates[0]
    for candidate in sampled_split_candidates:
        curr_loss = partition_loss_by(inputs, candidate)
        if curr_loss < min_loss:
            min_loss = curr_loss
            best_candidate = candidate
    split_candidates.remove(best_candidate)
    partion = partition_by(inputs, best_candidate)
    if len(partion[0]) == 0:
        return (best_candidate, {0: avg_days_till_loan, 
                                 1: build_tree(partion[1], num_levels-1, split_candidates, num_split_candidates)})
    if len(partion[1]) == 0:
        return (best_candidate, {0: build_tree(partion[0], num_levels-1, split_candidates, num_split_candidates),
                                 1: avg_days_till_loan})
    return (best_candidate, {0: build_tree(partion[0],num_levels-1,split_candidates, num_split_candidates), 
                             1: build_tree(partion[1],num_levels-1,split_candidates, num_split_candidates)})


"""************************************************************************
* function:  predict(tree, to_predict)
*
* arguments:
* 		-tree:  a tree built with the build_tree function
*		-to_predict:  a dictionary of features
*
* return value:  a value indicating a prediction of days_until_funded

* TODO:  Write a recursive function that uses "tree" and the values in the
*		 dictionary "to_predict" to output a predicted value.
************************************************************************"""
def predict(tree, to_predict):
    #checks if tree is a leaf node (end of tree)
    if type(tree) == np.float64:
        return tree
    
    #calls classify recursively on the relevant sub-tree
    attribute = tree[0]
    sub_tree = tree[1]
    
    if to_predict[ attribute ] == 0:
        return predict(sub_tree[0], to_predict)
    else:
        return predict(sub_tree[1], to_predict)
def forest_predict(trees, to_predict):
    ans = 0
    for tree in trees:
        ans += predict(tree, to_predict)
    return ans / len(trees)


"""************************************************************************
* function:  load_data()
* arguments:  N/A
* return value:  a list of tuples representing the loans data
* 
* TODO:  Read in the loans data from the provided csv file.  Store the
* 		 observations as a list of tuples (a, b), where 'a' is a dictionary
*		 of features and 'b' is the value of the days_until_funded variable
************************************************************************"""
def load_data():
    file_reader = open("loans_B_unlabeled.csv", "rt", encoding="utf8")
    data_dict  = csv.DictReader(file_reader)
    
    loans_data = []

    for row in data_dict:
        days_until_funded = row.pop("days_until_funded")
        loans_data.append((row, days_until_funded))
    
    return loans_data

def bootstrap_sample(inputs, length):
    if length >= len(inputs):
        length = len(inputs)
    idx_list = []
    for x in range(length):
        idx_list.append(random.randint(0, len(inputs)-1))
    sample = [inputs[x] for x in idx_list]
    return sample
        

def country_to_continent(country_name):
    country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    return country_continent_name

def process_data(file_name):
    print(file_name)
    # f = open("loans_B_unlabeled.csv", "rt", encoding="utf8")
    f = open(file_name, "rt", encoding="utf8")
    # positive_words = ['children' , 'community' , 'repayment' , 'food', 'pass', 'home' ,
    # 'fruit' , 'married' , 'morning' , 'born' ]
    # negative_words = ['school', 'old' , 'business' , 'village' , 
    # 'years', 'loan', 'additional', 'costly' , 'invest', 'expand']
    positive_words = ['years', 'children', 'married', 'help', 'lives', 'income', 'old', 'husband', 'living', 'selling', 'kiva']
    negative_words = ['loan', 'business', 'buy', 'family', 'work', 'house', 'store', 'improve']
    
    data = pd.read_csv(f)
    data['is_female'] = data['gender'].apply(lambda x: 1 if x == 'F' else 0)
    data['low_loan_amount'] = data['loan_amount'].apply(lambda x: 1 if x <= 300.0 else 0)
    data['high_loan_amount'] = data['loan_amount'].apply(lambda x: 1 if x >= 975.0 else 0)
    data['fast_repayment'] = data['repayment_term'].apply(lambda x: 1 if x <= 8 else 0)
    data['slow_repayment'] = data['repayment_term'].apply(lambda x: 1 if x >= 13 else 0)
    data['multi_lang'] = data['languages'].apply(lambda x: 1 if len(x.split('|')) > 2 else 0) 
    # data['is_peru'] = data['country'].apply(lambda x: 1 if x == 'Peru' else 0)
    data['slow_sector'] = data['sector'].apply(lambda x: 1 if x == 'Housing' else 0)
    data['fast_sector'] = data['sector'].apply(lambda x: 1 if x == 'Food' or 'Education' or 'Arts' or 'Agriculture' else 0)
    # data['is_philippines'] = data['country'].apply(lambda x: 1 if x == 'Philippines' else 0)
    # data['is_kenya'] = data['country'].apply(lambda x: 1 if x == 'Kenya' else 0)
    # data['is_nicaragua'] = data['country'].apply(lambda x: 1 if x == 'Nicaragua' else 0)
    # data['asian'] = data['country'].apply(lambda x: 1 if country_to_continent(x) == 'Asia' else 0)
    # data['north_american'] = data['country'].apply(lambda x: 1 if country_to_continent(x) == 'North America' else 0)
    data['contains_old'] = data['description'].apply(lambda x: 1 if  'old' in x.lower() else 0)
    data['contains_improve'] = data['description'].apply(lambda x: 1 if  'improve' in x.lower() else 0)
    data['contains_help'] = data['description'].apply(lambda x: 1 if  'help' in x.lower() else 0)
    data['contains_buy'] = data['description'].apply(lambda x: 1 if  'buy' in x.lower() else 0)
    data['contains_loan'] = data['description'].apply(lambda x: 1 if  'loan' in x.lower() else 0)
    data['asian'] = pd.NaT
    data['north_american'] = pd.NaT
        
    data['month'] = pd.DatetimeIndex(data['posted_date']).month
    data['is_early'] = data['month'].apply(lambda x: 1 if x == 1 or 2 or 3 else 0)
    data['is_late'] = data['month'].apply(lambda x: 1 if x == 6 or 11 or 12 else 0)

    #non-binary don't use
    # data['words'] = data['description'].apply(lambda x: x.lower().split())
    # data['negative_sentiment_other'] = pd.DataFrame(data['words'].tolist(),index=data.index).isin(negative_words).sum(1) 
    # data['positive_sentiment_other'] = pd.DataFrame(data['words'].tolist(),index=data.index).isin(positive_words).sum(1)
    #non-binary don't use
    
    print('parsing')
    
    for n in negative_words:
        data[n] = data['description'].str.contains(n, case=False).astype(int)
    data['negative_sentiment'] = data[negative_words].sum(axis=1)

    for p in positive_words:
        data[p] = data['description'].str.contains(p, case=False).astype(int) 
    data['positive_sentiment'] = data[positive_words].sum(axis=1)

    data['overall_sentiment'] = (data['positive_sentiment'] > data['negative_sentiment']) * 1 # cast to int
    data['high_negative_sentiment'] = data['negative_sentiment'].apply(lambda x : 1 if x > 4 else 0)
    data['high_positive_sentiment'] = data['positive_sentiment'].apply(lambda x : 1 if x > 4  else 0)
    # data = data.drop('days_until_funded', axis=1)
    # data = data.drop('words', axis=1)
    
    for index, row in data.iterrows():
        country = row['country']
        country_africa = ['congo', 'cote']
        country_asia = ['timor', 'myanmar', 'lao']
        if any(substring in country.lower() for substring in country_africa):
            data.at[index, 'african'] = 1
            data.at[index, 'american'] = 0
            data.at[index, 'european'] = 0
        elif any(substring in country.lower() for substring in country_asia):
            data.at[index, 'african'] = 0
            data.at[index, 'american'] = 0
            data.at[index, 'european'] = 0
        else: 
            continent = country_to_continent(country)
            if continent == 'Africa':
                data.at[index, 'african'] = 1
                data.at[index, 'american'] = 0
                data.at[index, 'european'] = 0
            elif continent == 'North America' or continent == 'South America':
                data.at[index, 'african'] = 0
                data.at[index, 'american'] = 1
                data.at[index, 'european'] = 0
            elif continent == 'Europe':
                data.at[index, 'african'] = 0
                data.at[index, 'american'] = 0
                data.at[index, 'european'] = 1
            else: 
                data.at[index, 'african'] = 0
                data.at[index, 'american'] = 0
                data.at[index, 'european'] = 0
    
    if file_name == 'loans_A_labeled.csv' or file_name == 'loans_AB_labeled.csv':
        data = data.to_dict('records')
        lis = []
        for ele in data:
            temp = ele['days_until_funded']
            del ele['days_until_funded']
            lis.append((ele,temp))
        return lis
    else:
        data = data.to_dict('records')
        print('dict')
        return data
        
    # return data.to_csv("loans_B_unlabed_plus.csv")

# %%
all_candidates = ['is_female', 'low_loan_amount', 'high_loan_amount', 'fast_repayment', 'slow_repayment',
                  'slow_sector', 'fast_sector', 'african', 'american', 'european', 
                  'is_early', 'is_late', 'multi_lang']
# all_candidates = ['overall_sentiment', 'is_female', 'low_loan_amount', 'fast_repayment', 'slow_repayment', 'is_peru'
#     ,'housing_sector', 'is_philippines', 'is_kenya', 'is_nicaragua', 'contains_old', 'contains_improve', 'contains_help',
#     'contains_buy', 'contains_loan', 'high_negative_sentiment', 'high_positive_sentiment']


'''k is layers
split_candidates is attributes you can change
length is the number of samples to choose for the bootstrap
num_trees is number of trees to make for the random forest
num_split_candidates is the number of candidates to randomly consider at each level
'''
def main(k, split_candidates, length = 10, num_trees = 10, num_split_candidates = 5):
    predictions = []
    # days = []
    loans = process_data("loans_AB_labeled.csv")
    trees = []
    for i in range(num_trees):
        new_loan = bootstrap_sample(loans, length)
        trees.append(build_tree(new_loan, k, split_candidates, num_split_candidates))
    acc = 0
    for i in range(len(loans)):
        # days.append(loans[i][1])
        p = forest_predict(trees, loans[i][0])
        predictions.append(p)
        acc += ((p - loans[i][1])**2)/len(loans)
    acc = np.round(acc, 2)
    loans_unlabeled = process_data("loans_B_unlabed_plus.csv")
    predicted = pd.DataFrame(columns=['ID', 'days_until_funded_CC_WG_AR'])
    for i in range(len(loans_unlabeled)):
        # if (i % 1000) == 0:
            # print(i)
        predicted.at[i, 'ID'] = loans_unlabeled[i]['id']
        prediction = forest_predict(trees, loans_unlabeled[i])
        predicted.at[i, 'days_until_funded_CC_WG_AR'] = prediction
    
    predicted.to_csv('loans_A2_predicted_CC_WG_AR.csv', encoding='utf-8', index=False)
    # predictions = np.array(predictions)
    # days = np.array(days)
    # print(np.unique(predictions))
    # print(len(np.unique(predictions)))
    # print(np.unique(days))
    # print(len(np.unique(days)))
    return (acc)

    # return process_data()
    # loans = load_data()
    # return loans
    # tree = build_tree(loans, 1)
    
    # correct_predictions = 0
    # for i in range(len(mushrooms)):
    #     if classify(tree, mushrooms[i][0]) == mushrooms[i][1]:
    #         correct_predictions += 1
            
    # return ((correct_predictions / float(len(mushrooms))) * 100)
    # change this to MSE -- sum across (actual - pred)**2 / numObs

# for i in range(1,10):
#     print(main(i, all_candidates))
# %%
# TREE 1: tree with 1 level
# print(main(1, all_candidates))

# %%
# TREE 2: tree with 3 levels
# main(3, all_candidates)

# # %%
# # TREE 2: tree with modified split_candidates list  
core_candidates = ['is_female', 'low_loan_amount', 'high_loan_amount', 'fast_repayment', 'slow_repayment']

# gave tree 4 levels bc only have 4 attributes in list
# main(4, all_candidates)
# %%
descriptions_only = ['overall_sentiment', 'contains_old', 'contains_improve', 'contains_help',
    'contains_buy', 'contains_loan', 'high_negative_sentiment', 'high_positive_sentiment']

everything = descriptions_only + all_candidates

print(main(5, everything, 100, 100, 10))
# gave tree 8 levels bc only have 8 attributes in list
# print(main(4, core_candidates))
# process_data('loans_B_unlabeled.csv')

# acc_list = []
# for i in range(1,21):
#  	acc_list.append(main(i, everything))

# num_levels = [x for x in range(1, 21)]

# plt.plot(num_levels, acc_list)
# plt.xticks(num_levels)
# plt.xlabel('number of levels')
# plt.ylabel('MSE')
# plt.title('Accuracy of Tree vs Number of Levels')

# print(acc_list)

# for count, (x,y) in enumerate(zip(num_levels, acc_list), start=1):
    
#     label = "{:.2f}".format(y)

#     if x == float(1):
#         plt.annotate(label, (x,y), textcoords="offset points", 
#                       xytext=(0,10), ha='center')
#         next
        
#     if count % 2 == 0:
#         plt.annotate(label, (x,y), textcoords="offset points", 
#                       xytext=(0,10), ha='center')



# # %%

# %%

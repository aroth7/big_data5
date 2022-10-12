# %%

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log
from collections import defaultdict, Counter
import copy


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
        
        h = (length / num_obs) * (squared_sum / length)
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
def build_tree(inputs, num_levels, candidates):
    split_candidates = copy.deepcopy(candidates) #make sure you can split on the same attribute in different branches
    # print(len(split_candidates), num_levels)
    days_till_loan_lst = [row[1] for row in inputs]
    days_till_loan = np.array(days_till_loan_lst)
    avg_days_till_loan = np.mean(days_till_loan)

    # if every row has the same number of days then stop splitting
    if avg_days_till_loan == days_till_loan_lst[0]:
        return avg_days_till_loan
    # print(split_candidates)
    if num_levels == 0 or len(split_candidates) == 0:
        
        return avg_days_till_loan
    min_loss = partition_loss_by(inputs, split_candidates[0])
    best_candidate = split_candidates[0]
    for candidate in split_candidates:
        curr_loss = partition_loss_by(inputs, candidate)
        if curr_loss < min_loss:
            min_loss = curr_loss
            best_candidate = candidate
    split_candidates.remove(best_candidate)
    partion = partition_by(inputs, best_candidate)
    if len(partion[0]) == 0:
        return (best_candidate, {1: build_tree(partion[1], num_levels-1, split_candidates)})
    if len(partion[1]) == 0:
        return (best_candidate, {0: build_tree(partion[0], num_levels-1, split_candidates)})
    return (best_candidate, {0: build_tree(partion[0],num_levels-1,split_candidates), 1: build_tree(partion[1],num_levels-1,split_candidates)})


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
    
    """TODO: change for non-binary"""
    if to_predict[ attribute ] == 0:
        return predict(sub_tree[0], to_predict)
    else:
        return predict(sub_tree[1], to_predict)


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
    file_reader = open("loans_A_labeled.csv", "rt", encoding="utf8")
    data_dict  = csv.DictReader(file_reader)
    
    loans_data = []

    for row in data_dict:
        days_until_funded = row.pop("days_until_funded")
        loans_data.append((row, days_until_funded))
    
    return loans_data

def process_data():
    # f = open("loans_B_unlabeled.csv", "rt", encoding="utf8")
    f = open("loans_A_labeled.csv", "rt", encoding="utf8")
    positive_words = ['children' , 'community' , 'repayment' , 'food', 'pass', 'home' ,
    'fruit' , 'married' , 'morning' , 'born' ]
    negative_words = ['school', 'old' , 'business' , 'village' , 
    'years', 'loan', 'additional', 'costly' , 'invest', 'expand']
    data = pd.read_csv(f)
    data['is_female'] = data['gender'].apply(lambda x: 1 if x == 'F' else 0)
    data['low_loan_amount'] = data['loan_amount'].apply(lambda x: 1 if x <= 375.0 else 0)
    data['fast_repayment'] = data['repayment_term'].apply(lambda x: 1 if x <= 8 else 0)
    data['slow_repayment'] = data['repayment_term'].apply(lambda x: 1 if x >= 13 else 0)
    data['is_peru'] = data['country'].apply(lambda x: 1 if x == 'Peru' else 0)
    data['housing_sector'] = data['sector'].apply(lambda x: 1 if x == 'Housing' else 0)
    data['is_philippines'] = data['country'].apply(lambda x: 1 if x == 'Philippines' else 0)
    data['is_kenya'] = data['country'].apply(lambda x: 1 if x == 'Kenya' else 0)
    data['is_nicaragua'] = data['country'].apply(lambda x: 1 if x == 'Nicaragua' else 0)
    data['contains_old'] = data['description'].apply(lambda x: 1 if  'old' in x.lower() else 0)
    data['contains_improve'] = data['description'].apply(lambda x: 1 if  'improve' in x.lower() else 0)
    data['contains_help'] = data['description'].apply(lambda x: 1 if  'help' in x.lower() else 0)
    data['contains_family'] = data['description'].apply(lambda x: 1 if  'family' in x.lower() else 0)
    data['contains_business'] = data['description'].apply(lambda x: 1 if  'business' in x.lower() else 0)

    #non-binary don't use
    data['words'] = data['description'].apply(lambda x: x.lower().split())
    data['negative_sentiment'] = pd.DataFrame(data['words'].tolist(),index=data.index).isin(negative_words).sum(1) 
    data['positive_sentiment'] = pd.DataFrame(data['words'].tolist(),index=data.index).isin(positive_words).sum(1)
    #non-binary don't use

    data['overall_sentiment'] = (data['positive_sentiment'] > data['negative_sentiment']) * 1 # cast to int
    data['high_negative_sentiment'] = data['negative_sentiment'].apply(lambda x : 1 if x > 3 else 0)
    data['high_positive_sentiment'] = data['positive_sentiment'].apply(lambda x : 1 if x > 3 else 0)
    # data = data.drop('days_until_funded', axis=1)
    # data = data.drop('words', axis=1)
    data = data.to_dict('records')
    lis = []
    for ele in data:
        temp = ele['days_until_funded']
        del ele['days_until_funded']
        lis.append((ele,temp))
    return lis
    # return data.to_csv("loans_B_unlabed_plus.csv")

# %%
all_candidates = ['overall_sentiment', 'is_female', 'low_loan_amount', 'fast_repayment', 'slow_repayment', 'is_peru'
    ,'housing_sector', 'is_philippines', 'is_kenya', 'is_nicaragua', 'contains_old', 'contains_improve', 'contains_help',
    'contains_family', 'contains_business', 'high_negative_sentiment', 'high_positive_sentiment']

def main(k, split_candidates):
    loans = process_data()
    tree = build_tree(loans, k, split_candidates)
    # print(tree)
    acc = 0
    for i in range(len(loans)):
        acc += ((predict(tree, loans[i][0]) - loans[i][1])**2)/len(loans)
    return ("MSE is: " + str(acc))

    # use below return statement when graphing
    # return acc 

# %%
# TREE 1: tree with 1 level
main(1, all_candidates)

# %%
# TREE 2: tree with 3 levels
main(3, all_candidates)

# %%
# TREE 2: tree with modified split_candidates list  
core_candidates = ['is_female', 'low_loan_amount', 'fast_repayment', 'slow_repayment']

# gave tree 4 levels bc only have 4 attributes in list
main(4, core_candidates)
# %%
descriptions_only = ['overall_sentiment', 'contains_old', 'contains_improve', 'contains_help',
    'contains_family', 'contains_business', 'high_negative_sentiment', 'high_positive_sentiment']

# gave tree 8 levels bc only have 8 attributes in list
main(8, descriptions_only)

acc_list = []
for i in range(1,9):
	acc_list.append(main(i, descriptions_only))

num_levels = [x for x in range(1, 9)]

plt.plot(num_levels, acc_list)
plt.xticks(num_levels)
plt.xlabel('number of levels')
plt.ylabel('MSE')
plt.title('MSE of Tree vs Number of Levels')



# %%
# TREE 5: running with 17 levels on all 17 candidates 
main(17, all_candidates)

acc_list = []
for i in range(1,18):
	acc_list.append(main(i, all_candidates))

num_levels = [x for x in range(1, 18)]

plt.plot(num_levels, acc_list)
plt.xticks(num_levels)
plt.xlabel('number of levels')
plt.ylabel('MSE')
plt.title('MSE of Tree vs Number of Levels')
# %%

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log

file_reader = open("loans_A2_labeled.csv", "rt", encoding="utf8")
data_dict  = csv.DictReader(file_reader)

loans_data = []

for row in data_dict:
    id_number = row.pop("id")
    days_until_funded = row.pop("days_until_funded")
    loans_data.append((id_number, float(days_until_funded)))
    
file_reader_predicted = open("loans_A2_predicted_CC_WG_AR.csv", "rt", encoding="utf8")
data_dict_predicted  = csv.DictReader(file_reader_predicted)

predicted_data = []

for row in data_dict_predicted:
    id_number = row.pop("ID")
    days_until_funded = row.pop("days_until_funded_CC_WG_AR")
    predicted_data.append((id_number, float(days_until_funded)))

# days = []
# predictions = []

acc = 0
for i in range(len(loans_data)):
    if loans_data[i][0] != predicted_data[i][0]:
        print("error")
    # days.append(loans_data[i][1])
    # predictions.append(predicted_data[i][1])
    acc += ((predicted_data[i][1] - loans_data[i][1])**2)/len(loans_data)
acc = np.round(acc, 2)

print(acc)

# core model MSE = 101.86, training MSE = 62.34
# core model two-level MSE = 98.74, training MSE = 76.52
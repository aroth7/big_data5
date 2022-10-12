import csv
import string
from collections import Counter

def count_word_frequency():
    fileReader = open("loans_A_labeled.csv", "rt", encoding="utf8")
    csvReader  = csv.DictReader(fileReader)
    str = ""
    for dcObservation in csvReader:
        # if "en" in dcObservation["languages"]:
        str+=dcObservation["description"]
        
    str = str.lower()
    str = str.translate(str.maketrans('', '', string.punctuation))
    str = str.split()
    # unique_words = set(str)

    # for word in unique_words:
    #     print(word," ",str.count(word))
    counts = Counter(str)
    print(counts.most_common(70))

count_word_frequency()
positive_words = ['children' , 'community' , 'repayment' , 'food', 'pass', 'home' ,
'fruit' , 'married' , 'morning' , 'born' ]
negative_words = ['school', 'old' , 'business' , 'village' , 
'years', 'loan', 'additional', 'costly' , 'invest', 'expand']

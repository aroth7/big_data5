import csv
import random

fileReader = open("loans_A_labeled.csv", "r", encoding="utf8")
csvReader  = csv.reader(fileReader)

fileWriter = open("loans_A1_labeled_subset.csv", "w", encoding="utf8", newline='')
csvWriter  = csv.writer(fileWriter)

fileWriterA2 = open("loans_A2_labeled_subset.csv", "w", encoding="utf8", newline='')
csvWriterA2  = csv.writer(fileWriterA2)

acHeader = next(csvReader)
csvWriter.writerow(acHeader)
csvWriterA2.writerow(acHeader)

for acRow in csvReader:
    if random.random() <= 0.5:
        csvWriter.writerow(acRow)
    else:
        csvWriterA2.writerow(acRow)

fileReader.close()
fileWriter.close()
fileWriterA2.close()
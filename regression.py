import csv
import numpy as np
import scipy as sp
import scipy.stats
import numpy.linalg
import pycountry_convert as pc
from datetime import datetime

fileReader = open("loans_AB_labeled.csv", "rt", encoding="utf8")
csvReader  = csv.DictReader(fileReader)

afDaysFunded  = list()
afFemale  = list()
afLanguages = list()

afRepayment = list()
afFastRepay = list()
afSlowRepay = list()

afLoanAmount = list()
afHighLoan = list()
afLowLoan = list()

afSector = list()
afCountry = list()
afContinent = list()

# afKeyword = list()
# afSentiment = list()
# afHighSentiment = list()

afMonth = list()

totalDaysFunded = 0
numObs = 0

def country_to_continent(country_name):
    country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    return country_continent_name

for dcObservation in csvReader:
    totalDaysFunded += int(dcObservation["days_until_funded"])
    numObs += 1
    fDaysFunded = int(dcObservation["days_until_funded"])
    
    # if dcObservation["gender"] == "M":
    #     fFemale = 0
    # else:
    #     fFemale = 1
    
    # languages = len(dcObservation["languages"].split("|")) - 1
    # if languages > 1:
    #     fLanguages = 1
    # else:
    #     fLanguages = 0
        
    # fRepayment = int(dcObservation["repayment_term"])
    # if fRepayment <= 8:
    #     fFastRepay = 1
    # else:
    #     fFastRepay = 0
        
    # if fRepayment >= 13:
    #     fSlowRepay = 1
    # else:
    #     fSlowRepay = 0
    
    # fLoanAmount = float(dcObservation["loan_amount"])
    # if fLoanAmount >= 975.0:
    #     fHighLoan = 1
    # else:
    #     fHighLoan = 0
    
    # if fLoanAmount >= 975.0:
    #     fLowLoan = 1
    # else:
    #     fLowLoan = 0
        
    # sector = dcObservation["sector"]
    # if sector == "Housing":
    #     fSector = 1
    # else:
    #     fSector = 0
        
    country = dcObservation["country"]
    # if country == "Cambodia":
    #     fCountry = 1
    # else:
    #     fCountry = 0

    country_others = ['congo', 'cote']
    country_asia = ['timor', 'myanmar', 'lao']
    
    if any(substring in country.lower() for substring in country_others):
        fContinent = 0
    elif any(substring in country.lower() for substring in country_asia):
        fContinent = 1
    else:
        continent = country_to_continent(country)
        if continent == "Asia":
            fContinent = 1
        else:
            fContinent = 0
        
    if 'family' in dcObservation['description'].lower():
        fKeyword = 1
    else:
        fKeyword = 0
        
    # fSentiment = 0
    # negative_words = ['loan', 'business', 'buy', 'family', 'work', 'house', 
    #                   'store', 'improve']
    # positive_words = ['years', 'children', 'married', 'help', 'lives', 'income', 
    #                   'old', 'husband', 'living', 'selling', 'kiva']
    
    # for word in dcObservation['description'].lower().split():
    #     if word in negative_words:
    #         fSentiment -= 1
    #     # if word in positive_words:
    #     #     fSentiment += 1
    
    # if fSentiment < -2:
    #     fHighSentiment = 1
    # else:
    #     fHighSentiment = 0
    
    date = datetime.strptime(dcObservation['posted_date'], '%Y-%m-%dT%H:%M:%SZ').date()
    month = date.month
    if month == 12:
        fMonth = 1
    else:
        fMonth = 0
            
    afDaysFunded.append(fDaysFunded)
    # afFemale.append(fFemale)
    # afLanguages.append(fLanguages)
    # afRepayment.append(fRepayment)
    # afFastRepay.append(fFastRepay)
    # afSlowRepay.append(fSlowRepay)
    # afLoanAmount.append(fLoanAmount)
    # afHighLoan.append(fHighLoan)
    # afLowLoan.append(fLowLoan)
    # afSector.append(fSector)
    # afCountry.append(fCountry)
    afContinent.append(fContinent)
    # afKeyword.append(fKeyword)
    # afSentiment.append(fSentiment)
    # afHighSentiment.append(fHighSentiment)
    afMonth.append(fMonth)

fileReader.close()

afDaysFunded  = np.array( afDaysFunded )
# afFemale  = np.array( afFemale )
# afLanguages = np.array( afLanguages )
# afRepayment = np.array( afRepayment )
# afFastRepay = np.array( afFastRepay )
# afSlowRepay = np.array( afSlowRepay )
# afLoanAmount = np.array( afLoanAmount )
# afHighLoan = np.array( afHighLoan )
# afLowLoan = np.array( afLowLoan )
afSector = np.array( afSector )
# afCountry = np.array( afCountry )
afContinent = np.array( afContinent )
# afKeyword = np.array( afKeyword )
# afSentiment = np.array( afSentiment )
# afHighSentiment = np.array( afHighSentiment )
afMonth = np.array( afMonth )

print("Mean(Days_Funded): ", float(totalDaysFunded / numObs))
print("Std(Days_Funded): ", np.std(afDaysFunded), "\n")
print("Correlation between days until funded and month: ", 
      sp.stats.pearsonr(afDaysFunded, afMonth))
print("OLS Regression: ", 
      np.linalg.lstsq(np.vstack([afMonth, np.ones(len(afMonth))]).T, 
                      afDaysFunded, rcond=None), "\n")
# print("Correlation between days until funded and female: ", 
#       sp.stats.pearsonr(afDaysFunded, afFemale))
# print("OLS Regression: ", 
#       np.linalg.lstsq(np.vstack([afFemale, np.ones(len(afFemale))]).T, 
#                       afDaysFunded, rcond=None), "\n")
# print("Correlation between days until funded and more than one language: ", 
#       sp.stats.pearsonr(afDaysFunded, afLanguages))
# print("OLS Regression: ", 
#       np.linalg.lstsq(np.vstack([afLanguages, np.ones(len(afLanguages))]).T, 
#                       afDaysFunded, rcond=None), "\n")
# print("Correlation between days until funded and repayment term: ", 
#       sp.stats.pearsonr(afDaysFunded, afRepayment))
# print("OLS Regression: ", 
#       np.linalg.lstsq(np.vstack([afRepayment, np.ones(len(afRepayment))]).T, 
#                       afDaysFunded, rcond=None))
# print("Correlation between days until funded and fast repayment term: ", 
#       sp.stats.pearsonr(afDaysFunded, afFastRepay))
# print("OLS Regression: ", 
#       np.linalg.lstsq(np.vstack([afFastRepay, np.ones(len(afFastRepay))]).T, 
#                       afDaysFunded, rcond=None))
# print("Correlation between days until funded and slow repayment term: ", 
#       sp.stats.pearsonr(afDaysFunded, afSlowRepay))
# print("OLS Regression: ", 
#       np.linalg.lstsq(np.vstack([afSlowRepay, np.ones(len(afSlowRepay))]).T, 
#                       afDaysFunded, rcond=None), "\n")
# print("Correlation between days until funded and loan amount: ", 
#       sp.stats.pearsonr(afDaysFunded, afLoanAmount))
# print("OLS Regression: ", 
#       np.linalg.lstsq(np.vstack([afLoanAmount, np.ones(len(afLoanAmount))]).T, 
#                       afDaysFunded, rcond=None))
# print("Correlation between days until funded and high loan amount: ", 
#       sp.stats.pearsonr(afDaysFunded, afHighLoan))
# print("OLS Regression: ", 
#       np.linalg.lstsq(np.vstack([afHighLoan, np.ones(len(afHighLoan))]).T, 
#                       afDaysFunded, rcond=None))
# print("Correlation between days until funded and low loan amount: ", 
#       sp.stats.pearsonr(afDaysFunded, afLowLoan))
# print("OLS Regression: ", 
#       np.linalg.lstsq(np.vstack([afLowLoan, np.ones(len(afLowLoan))]).T, 
#                       afDaysFunded, rcond=None), "\n")
# print("Correlation between days until funded and housing sector: ", 
#       sp.stats.pearsonr(afDaysFunded, afSector))
# print("OLS Regression: ", 
#       np.linalg.lstsq(np.vstack([afSector, np.ones(len(afSector))]).T, 
#                       afDaysFunded, rcond=None), "\n")
# print("Correlation between days until funded and Pakistan: ", 
#       sp.stats.pearsonr(afDaysFunded, afCountry))
# print("OLS Regression: ", 
#       np.linalg.lstsq(np.vstack([afCountry, np.ones(len(afCountry))]).T, 
#                       afDaysFunded, rcond=None), "\n")
print("Correlation between days until funded and Africa: ", 
      sp.stats.pearsonr(afDaysFunded, afContinent))
print("OLS Regression: ", 
      np.linalg.lstsq(np.vstack([afContinent, np.ones(len(afContinent))]).T, 
                      afDaysFunded, rcond=None), "\n")
# print("Correlation between days until funded and keyword in the description: ", 
#       sp.stats.pearsonr(afDaysFunded, afKeyword))
# print("OLS Regression: ", 
#       np.linalg.lstsq(np.vstack([afKeyword, np.ones(len(afKeyword))]).T, 
#                       afDaysFunded, rcond=None), "\n")
# print("Correlation between days until funded and sentiment score: ", 
#       sp.stats.pearsonr(afDaysFunded, afSentiment))
# print("OLS Regression: ", 
#       np.linalg.lstsq(np.vstack([afSentiment, np.ones(len(afSentiment))]).T, 
#                       afDaysFunded, rcond=None), "\n")
# print("Correlation between days until funded and high sentiment score: ", 
#       sp.stats.pearsonr(afDaysFunded, afHighSentiment))
# print("OLS Regression: ", 
#       np.linalg.lstsq(np.vstack([afHighSentiment, np.ones(len(afHighSentiment))]).T, 
#                       afDaysFunded, rcond=None), "\n")
# print("hi")
# print(afSentiment)
# print(np.percentile(afSentiment, [25, 50, 75]))
# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import datetime

#avg days funded for sector, repayment_term

# %%
f = open("loans_A_labeled.csv", "rt", encoding="utf8")
data = pd.read_csv(f)

# NOTE: cols = ['id', 'name', 'gender', 'pictured', 'description', 'loan_amount',
#        'activity', 'sector', 'country', 'town', 'posted_date',
#        'repayment_term', 'languages', 'days_until_funded']


# %%
def plot_scatter(x_col):
    plt.scatter(data[x_col], data['days_until_funded'])
    plt.xlabel(x_col)
    plt.ylabel("days until funded")
    plt.xticks(rotation = 90)
    plt.title("days_until_funded vs " + x_col)

# data['posted_date'] = pd.to_datetime(data['posted_date'], format='%Y-%m-%d')
# data['month_day'] = data['posted_date'].dt.strftime('%m-%d')
data['month'] = pd.DatetimeIndex(data['posted_date']).month
# sns.scatterplot(data=data, x="month", y="days_until_funded")
# plt.title("days_until_funded vs month of posted_date")
# plt.ylabel("days until funded")
# plt.xlabel('posted year')

months = data['month'].unique()
avg_days_per_month = {}

for month in months:
    avg_days_per_month[month] = data[data["month"] == month]["days_until_funded"].mean()

ticks = [x for x in range(1, 12)]

plt.bar(x=avg_days_per_month.keys(), height=avg_days_per_month.values())
plt.xticks(ticks= ticks, rotation = 90) 
plt.title("Average days until funded by month of posted_date")
plt.ylabel("days until funded")

# %%
# bar for gender
sns.countplot(data=data, x="days_until_funded", hue="gender")
plt.title("count of days funded by gender")

# %%

sns.scatterplot(data=data, x="loan_amount", y="days_until_funded", hue='sector')
plt.title("days_until_funded vs loan_amount by sector")
plt.ylabel("days until funded")
plt.xlabel('loan amount')

# %%
countries = data['country'].unique()

mini_df = pd.DataFrame()
mini_df['country'] = countries

avg_days_per_country = {}
avg_days_per_country_lst = []
for country in countries:
    avg_days_per_country_lst.append(data[data["country"] == country]["days_until_funded"].mean())
    avg_days_per_country[country] = data[data["country"] == country]["days_until_funded"].mean()

avg_amt_per_country = {}
avg_amt_per_country_lst = []
for country in countries:
    avg_amt_per_country_lst.append(data[data["country"] == country]["loan_amount"].mean())
    avg_amt_per_country[country] = data[data["country"] == country]["loan_amount"].mean()

mini_df["avg_amt"] = avg_amt_per_country_lst
mini_df["avg_days"] = avg_days_per_country_lst

# fig, ax = plt.bar(mini_df['country'], mini_df['avg_days'])
# plt.xticks(rotation = 90) 
# plt.ylabel("avg days until funded")

# ax2 = ax.twinx()

# plt.bar(mini_df['country'], mini_df['avg_amt'])
# plt.xticks(rotation = 90) 
# plt.ylabel("avg loan amount")

fig, ax = plt.subplots()
sns.barplot(data=mini_df, x="country", y="avg_days", ax=ax)
plt.xticks(rotation = 90) 
plt.ylabel("avg days until funded")

ax2 = plt.twinx()

sns.barplot(data=mini_df, x="country", y="avg_amt", ax=ax2)
plt.xticks(rotation = 90) 
plt.ylabel("avg loan amount")

#avg days per month from posted days





# plt.bar(x=avg_days_per_country.keys(), 
#         height=avg_days_per_country.values())
# plt.xticks(rotation = 90) 
# plt.title("Average days until funded by country")
# plt.ylabel("days until funded")

# ax2 = ax.twinx()
# ax2.bar(x=avg_amt_per_country.keys(), height=avg_amt_per_country.values(), ax=ax2)
# plt.title("Average days until funded by country")
# ax2.ylabel("days until funded")

# %%
sectors = data['sector'].unique()
avg_days_per_sector = {}

for sector in sectors:
    avg_days_per_sector[sector] = data[data["sector"] == sector]["days_until_funded"].mean()

plt.bar(x=avg_days_per_sector.keys(), height=avg_days_per_sector.values())
plt.xticks(rotation = 90) 
plt.title("Average days until funded by sector")
plt.ylabel("days until funded")

# %%
terms = data['repayment_term'].unique()
avg_days_per_term = {}

for term in terms:
    avg_days_per_term[term] = data[data["repayment_term"] == term]["days_until_funded"].mean()

ticks = [x for x in range(1, 28)]

plt.bar(x=avg_days_per_term.keys(), height=avg_days_per_term.values())
plt.xticks(ticks= ticks, rotation = 90) 
plt.title("Average days until funded by repayment_term")
plt.ylabel("days until funded")
# %%
num_langs = []

for idx, row in data.iterrows():
    lang_list = row['languages'].split("|")
    num_langs.append(len(lang_list))

num_with_1 = num_langs.count(2)
num_with_2 = num_langs.count(3)

num_langs_scaled = []

for item in num_langs:
    if item == 2:
        num_langs_scaled.append(item/num_with_1)
    else:
        num_langs_scaled.append(item/num_with_2)
    
data['num_languages'] = num_langs
data['num_languages_scaled'] = num_langs_scaled

sns.barplot(data=data, x="days_until_funded", y="num_languages_scaled")
plt.title("count of days funded by number of languages")
plt.legend(title='Number of Languages', labels=['1', '2'])
# %%

data.value_counts("num_languages", normalize=True).pipe(sns.catplot(data='data', x='days_until_funded', kind='bar'))

import pandas as pd
from pandas.api.types import CategoricalDtype




################################## ADULT DATASET ##################################################

df = pd.read_csv('Raw_data/adult/adult_train.csv', header= None)
mapping = {' <=50K': 0, ' >50K': 1}
df = df.replace({14: mapping})
df[1] = df[1].replace(' ?', df[1].mode()[0])
df[6] = df[6].replace(' ?', df[6].mode()[0])
df[13] = df[13].replace(' ?', df[13].mode()[0])


item = list(df[1].unique())
cat_type = CategoricalDtype(categories=item, ordered=True)
df[1] = df[1].astype(cat_type).cat.codes

item = list(df[6].unique())
cat_type = CategoricalDtype(categories=item, ordered=True)
df[6] = df[6].astype(cat_type).cat.codes

item = list(df[13].unique())
cat_type = CategoricalDtype(categories=item, ordered=True)
df[13] = df[13].astype(cat_type).cat.codes

df[3] = df[3].astype(CategoricalDtype(categories=df[3].unique(), ordered=True)).cat.codes
df[5] = df[5].astype(CategoricalDtype(categories=df[5].unique(), ordered=True)).cat.codes
df[7] = df[7].astype(CategoricalDtype(categories=df[7].unique(), ordered=True)).cat.codes
df[8] = df[8].astype(CategoricalDtype(categories=df[8].unique(), ordered=True)).cat.codes
df[9] = df[9].astype(CategoricalDtype(categories=df[9].unique(), ordered=True)).cat.codes


## Columns 3 and 4 are redundant ####
del df[4]
df.to_csv('adult_train.csv', index=False)


df = pd.read_csv('Raw_data/adult/adult.csv', header=None)
df = df.dropna(axis=1, how='all')
df.columns = range(15)
mapping = {'<=50K.': 0, '>50K.': 1}
df = df.replace({14: mapping})
df[1] = df[1].replace('?', df[1].mode()[0])
df[6] = df[6].replace('?', df[6].mode()[0])
df[13] = df[13].replace('?', df[13].mode()[0])

df[1] = df[1].astype(CategoricalDtype(categories=list(df[1].unique()), ordered=True)).cat.codes
df[6] = df[6].astype(CategoricalDtype(categories=list(df[6].unique()), ordered=True)).cat.codes
df[13] = df[13].astype(CategoricalDtype(categories=list(df[13].unique()), ordered=True)).cat.codes

df[3] = df[3].astype(CategoricalDtype(categories=df[3].unique(), ordered=True)).cat.codes
df[5] = df[5].astype(CategoricalDtype(categories=df[5].unique(), ordered=True)).cat.codes
df[7] = df[7].astype(CategoricalDtype(categories=df[7].unique(), ordered=True)).cat.codes
df[8] = df[8].astype(CategoricalDtype(categories=df[8].unique(), ordered=True)).cat.codes
df[9] = df[9].astype(CategoricalDtype(categories=df[9].unique(), ordered=True)).cat.codes
del df[4]
df.to_csv('adult_test.csv', index=False)
 
#########################################################################################################


#################################### COMPASS DATASET #####################################################
df = pd.read_csv('Raw_data/compass/propublica_data_for_fairml.csv')
df1 = df[(df['African_American'] == 1).values | ((df['African_American']==0).values & (df['Asian']==0).values & (df['Hispanic']==0).values
 	& (df['Other']==0).values) & (df['Native_American']==0).values]

data = df1[['Number_of_Priors', 'score_factor','Age_Above_FourtyFive', 'Age_Below_TwentyFive', 'African_American',
	'Female','Misdemeanor', 'Two_yr_Recidivism']]

data.to_csv('compass.csv')

#################################### CRIMES DATASET #####################################################
df = pd.read_csv('Raw_data/crimedata.csv',  encoding = "ISO-8859-1")
lis = list(df.columns)
df[lis[-1]] = df[lis[-1]].replace('?', 0)
df[lis[-2]] = df[lis[-2]].replace('?', 0)
df['label'] = df[lis[-1]].values.astype('float') + df[lis[-2]].values.astype('float')
m = df['label'].mean()
df.loc[df['label'] < m, 'label'] = 0
df.loc[df['label'] >= m, 'label'] = 1
lis = lis[1:] + ['label']
df = df[lis]
df['race'] = 0
df.loc[df['racepctblack'] < df['racePctWhite'], 'race'] = 1
df.loc[df['racepctblack'] >= df['racePctWhite'], 'race'] = 0
df.loc[df['racePctAsian'] >= df['racePctWhite'], 'race'] = 0
df.loc[df['racePctHisp'] >= df['racePctWhite'], 'race'] = 0 
for i in lis:
	df[i] = df[i].replace('?', df[i].mode()[0])
df[lis[0]] = df[lis[0]].astype(CategoricalDtype(categories=df[lis[0]].unique(), ordered=True)).cat.codes
df[lis[1]] = df[lis[1]].astype(CategoricalDtype(categories=df[lis[1]].unique(), ordered=True)).cat.codes
df[lis[2]] = df[lis[2]].astype(CategoricalDtype(categories=df[lis[2]].unique(), ordered=True)).cat.codes
lis = lis[:-1] + ['race'] + [lis[-1]]
df = df[lis]
for i in range(102,145):
	if df[lis[i]].dtype != 'O':
		continue
	else:
		df.loc[df[lis[i]] == '?', lis[i]]=0
for i in lis:
	df[i] = pd.to_numeric(df[i])

df.to_csv('crimes.csv', index=False)

#################################### BANK DATASET #######################################################
df = pd.read_csv('Raw_data/bank-additional/bank-additional-full.csv', delimiter=';')
c = list(df.columns)
for i in c:
	if df[i].dtype != 'int64' and df[i].dtype != 'float64' :
		df[i] = df[i].astype(CategoricalDtype(categories=df[i].unique(), ordered=True)).cat.codes
labels = ["{0} - {1}".format(i, j) for i,j in [[0,25],[25,65],[65,100]]]
df['age_group'] = pd.cut(df.age, [0, 25, 65, 100], right=False, labels=labels)
mapping = {'0 - 25': 0, '65 - 100': 0, '25 - 65': 1}
df = df.replace({'age_group': mapping})
cols = list(df.columns)
cols = [cols[-1]] + cols[0:-1]
df.columns = cols

df_train = df
df_train.to_csv('bank_train.csv', index=False)
#########################################################################################################


####### ProtectedVar = 8 #########
with open('Raw_data/german.data') as f:
	lines= f.readlines()
line = []
for l in lines:
	line.append(l.strip().split(' '))
df1 = pd.DataFrame(line)
df = df1.copy()
for i in range(1, 22):
	if i in [1,3,4,6,7,9,10,12,14,15,17,19,20]:
		df[i-1] = df[i-1].astype('category').cat.codes
	else:
		df[i-1] = df[i-1].astype('int64')

# gender
df1[8].replace(['A91'], 0, inplace=True)
df1[8].replace(['A92'], 1, inplace=True)
df1[8].replace(['A93'], 0, inplace=True)
df1[8].replace(['A94'], 0, inplace=True)
df[8] = df1[8]
df.to_csv('german.csv', index=False)

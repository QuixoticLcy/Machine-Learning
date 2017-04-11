import pandas as pd
import numpy as np

data1 = pd.read_csv('bots_data.csv',header = 0)
data2 = pd.read_csv('nonbots_data.csv',header = 0)
# append bots and nonbots and shuffle it
data = data1.append(data2)
from sklearn.utils import shuffle
data = shuffle(data)

# save is a dataframe to store usefull boolean attributes
save = pd.DataFrame()






# deal with numerical count attributes, classify them into 3 types: less than 10; between 10 and 1000; more than 1000
def classify_count(name):
    below_10 = []
    bettwen_10_1000 = []
    above_1000 = []
    for val in data[name]:
        below_10.append(1 if val <= 10 else 0)
        bettwen_10_1000.append(1 if val > 10 and val <= 1000 else 0)
        above_1000.append(1 if val > 1000 else 0)
    save[name,'below_10'] = below_10
    save[name,'bettwen_10_1000'] = bettwen_10_1000
    save[name,'above_1000'] = above_1000
classify_count('followers_count')
classify_count('friends_count')
classify_count('listedcount')
classify_count('favourites_count')
classify_count('statuses_count')
save.head()



# deal with boolean attributes, classify them into 2 types: true or false
def classify_true_false(name):
    store = []
    for val in data[name]:
        if val == True:
            store.append(1)
        else:
            store.append(0)
    save[name] = store
classify_true_false('verified')
classify_true_false('default_profile')
classify_true_false('default_profile_image')
classify_true_false('has_extended_profile')
save.head()



# deal with string attributes, classify them into 2 types: null or non-null
def classify_null(name):
    store = []
    for val in data[name]:
        if str(val) == 'nan' or len(str(val)) < 3:
            store.append(1)
        else:
            store.append(0)
    save[name] = store
classify_null('location')
classify_null('description')
save.head()



# deal with ratio attributes, classify them into 3 types: less than 0.5; between 0.5 and 2; more than 2
def classify_ratio(name1, name2):
    ratio_below_05 = []
    ratio_between_05_2 = []
    ratio_above_2 = []
    for index, row in data.iterrows():
        ratio = float(row[name1] + 1) / float(row[name2] + 1)
        ratio_below_05.append(1 if ratio <= 0.5 else 0)
        ratio_between_05_2.append(1 if ratio >= 0.5 and ratio <= 2 else 0)
        ratio_above_2.append(1 if ratio > 2 else 0)
    save[name1, name2,'ratio_below_05'] = ratio_below_05
    save[name1, name2,'ratio_between_05_2'] = ratio_between_05_2
    save[name1, name2,'ratio_above_2'] = ratio_above_2
classify_ratio('followers_count','friends_count')
classify_ratio('statuses_count','friends_count')
save.head()



# new a classifier
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()

y_data = pd.DataFrame()
y_data['bot'] = data['bot']

# train it and than test it for 10 times, get the average score
from sklearn.model_selection import train_test_split
score = []
for i in range(10):
    x_train,x_test,y_train,y_test = train_test_split(save,y_data,test_size=0.5)
    clf = clf.fit(x_train,y_train)
    score.append(clf.score(x_test,y_test))

print np.mean(score)


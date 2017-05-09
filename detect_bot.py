import pandas as pd
import numpy as np
import sklearn.neural_network
from sklearn.model_selection import cross_val_score



def update_field(name):
    store = []
    for val in data[name]:
        if str(val) == 'nan' or len(str(val)) < 3:
            store.append(0)
        else:
            store.append(1)
    save[name] = store

def update_tf(name):
    store = []
    for val in data[name]:
        # if len(store) > 1300:
        #     # print True
        if val == True:
            store.append(1)
        else:
            store.append(0)
    save[name] = store

def update_ratio(name_1,name_2):
    store = []

    for index, row in data.iterrows():
        ratio = float(row[name_1] + 1) / float(row[name_2] + 1)
        store.append(ratio)
    name = str(name_1) + str(name_2)
    save[name] = store

def contain_bot():
    store = []
    for index, row in data.iterrows():
        name = str(row['name']).lower()
        des = str(row['description']).lower()
        key_word = 'bot'
        if key_word in des or key_word in name:
            store.append(1)
        else:
            store.append(0)
    save['contain'] = store

def des_len():
    store = []
    for val in data['description']:
        if str(val).lower() == 'nan' or str(val).lower() =='none':
            store.append(0)
        else:
            store.append(len(val))
    save['des_len'] = store

save = pd.DataFrame()
data = pd.read_csv('training_data_2_csv_UTF.csv',header = 0)
save['followers_count'] = data['followers_count']
save['friends_count'] = data['friends_count']
save['listedcount'] = data['listedcount']
save['favourites_count'] = data['favourites_count']
save['statuses_count'] = data['statuses_count']



update_tf('verified')
update_ratio('followers_count','friends_count')
update_ratio('statuses_count','followers_count')
update_ratio('followers_count','listedcount')
update_ratio('listedcount','friends_count')
contain_bot()

y_data = pd.DataFrame()
y_data['bot'] = data['bot']


clf = sklearn.ensemble.GradientBoostingClassifier(loss='exponential',n_estimators=110,max_depth = 3,learning_rate=0.095)

clf.fit(save,y_data)
importances =  clf.feature_importances_
std = []

indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(save.shape[1]):
    print("%d. feature %d %s (%f)" % (f + 1, indices[f], list(save)[f], importances[indices[f]]))


print np.mean(cross_val_score(clf,save,y_data['bot'],cv=5,scoring='accuracy'))
print np.mean(cross_val_score(clf,save,y_data['bot'],cv=5,scoring='roc_auc'))
print np.mean(cross_val_score(clf,save,y_data['bot'],cv=5,scoring='f1'))
print np.mean(cross_val_score(clf,save,y_data['bot'],cv=5,scoring='recall'))
print np.mean(cross_val_score(clf,save,y_data['bot'],cv=5,scoring='precision'))

# clf.fit(save,y_data)
x_test = pd.DataFrame()
# y_test = pd.DataFrame()
data = pd.read_csv('test_data_4_students.csv',header=0)

def transfer_data_test(ori_name, des_name):
    store = []

    for val in data[ori_name]:
        if isinstance(val, float):
            store.append(0)
        elif val == 'nan' or val.lower() == 'none':
            store.append(0)
        else:
            store.append(int(val))
    x_test[des_name] = store

print(save.shape)

transfer_data_test('friends_count','friends_count')
transfer_data_test('listed_count','listedcount')
transfer_data_test('favorites_count','favourites_count')
transfer_data_test('statuses_count','statuses_count')
transfer_data_test('followers_count','followers_count')


def update_field_test(name):
    store = []
    for val in data[name]:
        if str(val) == 'nan' or str(val.lower()) == 'none' or len(str(val)) < 3:
            store.append(0)
        else:
            store.append(1)
    x_test[name] = store

def update_tf_test(name):
    store = []
    for val in data[name]:
        if isinstance(val, float):
            store.append(0)

        elif val == True or val.lower() == 'true':
            store.append(1)
        else:
            store.append(0)
    x_test[name] = store

def update_ratio_test(name_1,name_2):
    store = []

    for index, row in x_test.iterrows():

        ratio = float(row[name_1] + 1) / float(row[name_2] + 1)
        store.append(ratio)
    name = str(name_1) + str(name_2)
    x_test[name] = store


def contain_bot_test(data):
    store = []
    for index, row in data.iterrows():
        name = str(row['name']).lower()
        des = str(row['description']).lower()
        # name = re.split(' |\,|_|;',name)
        # des = re.split(' |\,|_|;',des)
        key_word = 'bot'
        if key_word in des or key_word in name:
            store.append(1)
        else:
            store.append(0)
    x_test['contain'] = store

def des_len_test():
    store = []
    for val in data['description']:
        if str(val).lower() == 'nan' or str(val).lower() =='none':
            store.append(0)
        else:
            store.append(len(val))
    x_test['des_len'] = store



update_tf_test('verified')
update_ratio_test('followers_count','friends_count')
# update_ratio_test('statuses_count','friends_count')
update_ratio_test('statuses_count','followers_count')
update_ratio_test('followers_count','listedcount')
update_ratio_test('listedcount','friends_count')
# update_ratio_test('listedcount','friends_count')
contain_bot_test(data)
x_test = x_test[:575]
y_test = clf.predict(x_test)

result = pd.DataFrame()
id_store =[]
for val in data['id'][:575]:
    id_store.append(int(val))

result['Id'] = id_store[:575]
result['Bot'] = y_test

result.to_csv('result.csv',sep=',',encoding='utf-8',index=False)

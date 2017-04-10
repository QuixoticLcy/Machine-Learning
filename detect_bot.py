import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
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
        if val == True:
            store.append(1)
        else:
            store.append(0)
    save[name] = store

save = pd.DataFrame()
data = pd.read_csv('whole.csv',header = 0)
save['followers_count'] = data['followers_count']
save['friends_count'] = data['friends_count']
save['listedcount'] = data['listedcount']
save['favourites_count'] = data['favourites_count']
save['statuses_count'] = data['statuses_count']
update_field('location')
update_field('description')
# print data['location']
# print "------------------------"
# print data['description']

update_tf('verified')
update_tf('default_profile')
update_tf('default_profile_image')
update_tf('has_extended_profile')

y_data = pd.DataFrame()
y_data['bot'] = data['bot']
# print y_data


clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True,
                intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
# print np.mean(cross_val_score(clf,save,y_data,cv= 5, scoring = 'f1'))
score = []
for i in range(20):
    x_train,x_test,y_train,y_test = train_test_split(save,y_data,test_size=0.2)
    clf = clf.fit(x_train,y_train)
    score.append(clf.score(x_test,y_test))

print np.mean(score)




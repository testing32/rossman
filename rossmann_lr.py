import os
import csv
from datetime import timedelta
import datetime
import pickle

import xgboost as xgb
import numpy as np
from sklearn import cross_validation, linear_model
from sklearn.metrics import make_scorer

# Constants
DATA_DIR = "/home/testing32/Rossmann_Data/"
#DATA_DIR = "rossmann_data/"
CLEANED_DATA_DIR = "cleaned_data/"
OUTPUT_DIR = "prediction/"

SIX_WEEKS = timedelta(days=42)

# Turns the day of the week digit into a feature list
def get_day_of_week_list(day):
    day_list = []
    
    for i in range(1, 8):
        if day == str(i):
            day_list.append(1)
        else:
            day_list.append(0)
            
    return day_list

# Turns an abc into a feature list
def get_abc_list(abc_char):
    abc_list = []
    
    for i in ("a", "b", "c"):
        if i == abc_char:
            abc_list.append(1)
        else:
            abc_list.append(0)
            
    return abc_list

"""
Gets a dictionary containing the store feature set
"""
def get_store_dict():
    store_dict = {}
    
    today = datetime.date.today()
    
    with open(DATA_DIR + 'store.csv', 'rb') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)  # skip the headers
        for row in reader:
            
            if row[5] == '':
                comp_open = today
            else:
                comp_open = datetime.date(int(row[5]), int(row[4]), today.day)
            
            feature_list = get_abc_list(row[1]) + get_abc_list(row[2])
            if row[3] == '':
                feature_list.append(50000)
            else:
                feature_list.append(int(row[3]))
            feature_list.append((today - comp_open).days)          
            
            store_dict[row[0]] = feature_list 
            
    return store_dict

"""
The training data is laid out like so:
Store ID, Day of week (1-Monday, 7-Sunday), Date, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday

The store data is laid out like so:
Store ID, StoreType, Assortment, CompetitionDistance, CompetitionOpenSinceMonth, CompOpenSinceYear, 
    Promo2, Promo2SinceWeek, PromoSinceYear, PromoInterval
"""
def clean_training_data():
    
    store_dict = get_store_dict()
    store_history = {}
    
    x = []
    y = []
    
    # First, get all of the training data out the file
    with open(DATA_DIR + 'train.csv', 'rb') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)  # skip the headers
        for row in reader:
            
            if row[0] not in store_history.keys():
                store_history[row[0]] = {'average_sales' : 0.0, 'average_customers': 0.0, 'items' : 0.0}
            
            # is store open
            if row[5] != '1':
                continue
                        
            store_history[row[0]][datetime.datetime.strptime(row[2], '%m-%d-%Y')] = row
            
            store_history[row[0]]['average_sales'] += float(row[3])
            store_history[row[0]]['average_customers'] += float(row[4])
            store_history[row[0]]['items'] += 1.0

    # Create our training data
    for hash_key in store_history.keys():
        
        average_sales = store_history[hash_key]['average_sales'] / store_history[hash_key]['items']
        average_customers = store_history[hash_key]['average_customers'] / store_history[hash_key]['items']
        
        for store_date in store_history[hash_key].keys():
            
            if isinstance(store_date, basestring):
                continue
            
            predict_data = store_history[hash_key][store_date]
            
            # create feature set, start with the store features
            feature_list = list(store_dict[hash_key])
            
            # add the day of the week
            feature_list += get_day_of_week_list(predict_data[1])
            
            # add date information
            feature_list.append(store_date.day)
            feature_list.append(store_date.month)
            feature_list.append(store_date.year)
            
            # is the store open
            #feature_list.append(int(predict_data[5]))
            # is it a promo
            feature_list.append(int(predict_data[6]))
            # is it a state holiday this has letters in it, need to figure that out
            #feature_list.append(int(predict_data[7]))
            # is it a school holiday
            feature_list.append(int(predict_data[8]))
            
            feature_list.append(average_sales)
            feature_list.append(average_customers)
            
            # sales data from 6 weeks ago
            #feature_list.append(store_history[hash_key][store_date - SIX_WEEKS][3])
            
            x.append(feature_list)
            
            # append sales 6 weeks out
            y.append(int(predict_data[3]))

    pickle.dump(x, open(CLEANED_DATA_DIR + "x.pkl", 'wb'))
    pickle.dump(y, open(CLEANED_DATA_DIR + "y.pkl", 'wb'))
    pickle.dump(store_history, open(CLEANED_DATA_DIR + "store_history.pkl", 'wb'))
    pickle.dump(store_dict, open(CLEANED_DATA_DIR + "store_dict.pkl", 'wb'))
    

def clean_test_data():
    x_test = []
    include_ids = []
    skipped_ids = []
    
    store_history = pickle.load(open(CLEANED_DATA_DIR + "store_history.pkl", 'rb'))
    store_dict = pickle.load(open(CLEANED_DATA_DIR + "store_dict.pkl", 'rb'))
    
    with open(DATA_DIR + 'test.csv', 'rb') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)  # skip the headers
        for row in reader:
            
            if row[4] == '0':
                skipped_ids.append(row[0])
                continue
            
            include_ids.append(row[0])
            
            # create feature set, start with the store features
            feature_list = list(store_dict[row[1]])
            
            # add the day of the week
            feature_list += get_day_of_week_list(row[2])
            
            # create the date
            store_date = datetime.datetime.strptime(row[3], '%Y-%m-%d')
            
            # add date information
            feature_list.append(store_date.day)
            feature_list.append(store_date.month)
            feature_list.append(store_date.year)
            
            # add promo
            feature_list.append(int(row[5]))
            
            # is it a school holiday
            feature_list.append(int(row[7]))
            
            average_sales = store_history[row[1]]['average_sales'] / store_history[row[1]]['items']
            average_customers = store_history[row[1]]['average_customers'] / store_history[row[1]]['items']
            
            # add average sales and customers
            feature_list.append(average_sales)
            feature_list.append(average_customers)
            
            x_test.append(feature_list)
            
    return x_test, include_ids, skipped_ids

def rmspe(ground_truth, predictions):
    import math
    #return math.sqrt(np.sum(np.square((ground_truth - predictions)/ground_truth))/float(len(predictions)))
    sums =  np.square((ground_truth - predictions)/ground_truth)
    sum = 0
    for value in sums:
        if value != float('inf'):
            sum += value
    sum = sum / float(len(predictions))
    return math.sqrt(sum)

def predict_linear_regression():
    
    x = pickle.load(open(CLEANED_DATA_DIR + "x.pkl", 'rb'))
    y = pickle.load(open(CLEANED_DATA_DIR + "y.pkl", 'rb'))

    loss  = make_scorer(rmspe, greater_is_better=False)

    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2, random_state=0)
    clf = linear_model.LinearRegression()
    clf.fit(x_train, y_train)
    print(loss(clf, x_test, y_test))
    """
    for i in range(0, len(y_test)):
        prediction = clf.predict(x_test[i])
        print("Prediction: " + str(prediction) + " Actual: " + str(y_test[i]))
    """
    #scores = cross_validation.cross_val_score(clf, x, y, cv=5, scoring="mean_squared_error")
    #print(scores)

def predict_actual_test():
    
    x = pickle.load(open(CLEANED_DATA_DIR + "x.pkl", 'rb'))
    y = pickle.load(open(CLEANED_DATA_DIR + "y.pkl", 'rb'))
    
    clf = linear_model.LinearRegression()
    clf.fit(x, y)
    
    x_test_items, include_ids, exclude_ids = clean_test_data()
    
    with open(OUTPUT_DIR + 'submission.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id','Sales'])
        for i in range(0, len(include_ids)):
            writer.writerow([include_ids[i], clf.predict(x_test_items[i])[0]])

        for i in range(0, len(exclude_ids)):
            writer.writerow([exclude_ids[i], 0])
        
if __name__ == '__main__':
    #clean_training_data()
    #predict_linear_regression()
    #predict_actual_test()
    predict_actual_test()
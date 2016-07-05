import pandas as pd
import numpy as np
from sklearn import cross_validation
import xgboost as xgb

DATA_DIR = "/home/testing32/Rossmann_Data/"
OUTPUT_DIR = "prediction/"

# Thanks to Chenglong Chen for providing this in the forum
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w


def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe


def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe

def get_store_state_dict():
    import csv
    state_dict = {}
    value = 0
    with open(DATA_DIR + 'store_states.csv', 'rb') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)  # skip the headers
        for row in reader:
            if row[1] not in state_dict.keys():
                state_dict[row[1]] = value
                value += 1
                
    return state_dict

# Gather some features
def build_features_train(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly
    features.extend(['Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                     'CompetitionOpenSinceYear', 'Promo', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear'])

    # add some more with a bit of preprocessing
    features.append('SchoolHoliday')
    data['SchoolHoliday'] = data['SchoolHoliday'].astype(float)
    #
    #features.append('StateHoliday')
    #data.loc[data['StateHoliday'] == 'a', 'StateHoliday'] = '1'
    #data.loc[data['StateHoliday'] == 'b', 'StateHoliday'] = '2'
    #data.loc[data['StateHoliday'] == 'c', 'StateHoliday'] = '3'
    #data['StateHoliday'] = data['StateHoliday'].astype(float)

    features.append('DayOfWeek')
    features.append('month')
    features.append('day')
    features.append('year')
    
    data['year'] = data.Date.apply(lambda x: x.split('-')[2])
    data['year'] = data['year'].astype(float)
    data['month'] = data.Date.apply(lambda x: x.split('-')[0])
    data['month'] = data['month'].astype(float)
    data['day'] = data.Date.apply(lambda x: x.split('-')[1])
    data['day'] = data['day'].astype(float)

    features.append('StoreType')
    data.loc[data['StoreType'] == 'a', 'StoreType'] = '1'
    data.loc[data['StoreType'] == 'b', 'StoreType'] = '2'
    data.loc[data['StoreType'] == 'c', 'StoreType'] = '3'
    data.loc[data['StoreType'] == 'd', 'StoreType'] = '4'
    data['StoreType'] = data['StoreType'].astype(float)

    features.append('Assortment')
    data.loc[data['Assortment'] == 'a', 'Assortment'] = '1'
    data.loc[data['Assortment'] == 'b', 'Assortment'] = '2'
    data.loc[data['Assortment'] == 'c', 'Assortment'] = '3'
    data['Assortment'] = data['Assortment'].astype(float)

# Gather some features
def build_features(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly
    features.extend(['Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                     'CompetitionOpenSinceYear', 'Promo', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear'])

    # add some more with a bit of preprocessing
    features.append('SchoolHoliday')
    data['SchoolHoliday'] = data['SchoolHoliday'].astype(float)
    #
    #features.append('StateHoliday')
    #data.loc[data['StateHoliday'] == 'a', 'StateHoliday'] = '1'
    #data.loc[data['StateHoliday'] == 'b', 'StateHoliday'] = '2'
    #data.loc[data['StateHoliday'] == 'c', 'StateHoliday'] = '3'
    #data['StateHoliday'] = data['StateHoliday'].astype(float)

    features.append('DayOfWeek')
    features.append('month')
    features.append('day')
    features.append('year')
    
    data['year'] = data.Date.apply(lambda x: x.split('-')[0])
    data['year'] = data['year'].astype(float)
    data['month'] = data.Date.apply(lambda x: x.split('-')[1])
    data['month'] = data['month'].astype(float)
    data['day'] = data.Date.apply(lambda x: x.split('-')[2])
    data['day'] = data['day'].astype(float)

    features.append('StoreType')
    data.loc[data['StoreType'] == 'a', 'StoreType'] = '1'
    data.loc[data['StoreType'] == 'b', 'StoreType'] = '2'
    data.loc[data['StoreType'] == 'c', 'StoreType'] = '3'
    data.loc[data['StoreType'] == 'd', 'StoreType'] = '4'
    data['StoreType'] = data['StoreType'].astype(float)

    features.append('Assortment')
    data.loc[data['Assortment'] == 'a', 'Assortment'] = '1'
    data.loc[data['Assortment'] == 'b', 'Assortment'] = '2'
    data.loc[data['Assortment'] == 'c', 'Assortment'] = '3'
    data['Assortment'] = data['Assortment'].astype(float)
    
    # Gather some features
def new_build_features_train(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly
    features.extend(['Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                     'CompetitionOpenSinceYear', 'Promo', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear'])

    # add some more with a bit of preprocessing
    features.append('SchoolHoliday')
    data['SchoolHoliday'] = data['SchoolHoliday'].astype(float)
    
    features.append('StateHolidayA')
    features.append('StateHolidayB')
    features.append('StateHolidayC')
    
    data['StateHolidayA'] = 0
    data['StateHolidayB'] = 0
    data['StateHolidayC'] = 0
    
    data.loc[data['StateHoliday'] == 'a', 'StateHolidayA'] = '1'
    data.loc[data['StateHoliday'] == 'b', 'StateHolidayB'] = '1'
    data.loc[data['StateHoliday'] == 'c', 'StateHolidayC'] = '1'
    
    data['StateHolidayA'] = data['StateHolidayA'].astype(float)
    data['StateHolidayB'] = data['StateHolidayB'].astype(float)
    data['StateHolidayC'] = data['StateHolidayC'].astype(float)

    features.append('DayOfWeek')
    features.append('month')
    features.append('day')
    features.append('year')
    
    data['year'] = data.Date.apply(lambda x: x.split('-')[2])
    data['year'] = data['year'].astype(float)
    data['month'] = data.Date.apply(lambda x: x.split('-')[0])
    data['month'] = data['month'].astype(float)
    data['day'] = data.Date.apply(lambda x: x.split('-')[1])
    data['day'] = data['day'].astype(float)

    features.append('StoreTypeA')
    features.append('StoreTypeB')
    features.append('StoreTypeC')
    
    data['StoreTypeA'] = 0
    data['StoreTypeB'] = 0
    data['StoreTypeC'] = 0
    
    data.loc[data['StoreType'] == 'a', 'StoreTypeA'] = '1'
    data.loc[data['StoreType'] == 'b', 'StoreTypeB'] = '1'
    data.loc[data['StoreType'] == 'c', 'StoreTypeC'] = '1'
    
    data['StoreTypeA'] = data['StoreTypeA'].astype(float)
    data['StoreTypeB'] = data['StoreTypeB'].astype(float)
    data['StoreTypeC'] = data['StoreTypeC'].astype(float)

    features.append('AssortmentA')
    features.append('AssortmentB')
    
    data['AssortmentA'] = 0
    data['AssortmentB'] = 0
    
    data.loc[data['Assortment'] == 'a', 'AssortmentA'] = '1'
    data.loc[data['Assortment'] == 'b', 'AssortmentB'] = '1'
    
    data['AssortmentA'] = data['AssortmentA'].astype(float)
    data['AssortmentB'] = data['AssortmentB'].astype(float)

    """ State doesn't help
    features.append('StoreState')
    state_dict = get_store_state_dict()
    data['StoreState'] = 0
    
    for key in state_dict.keys():
        data.loc[data['State'] == key, 'StoreDate'] = float(state_dict[key])
    """

# Gather some features
def new_build_features(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly
    features.extend(['Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                     'CompetitionOpenSinceYear', 'Promo', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear'])

    # add some more with a bit of preprocessing
    features.append('SchoolHoliday')
    data['SchoolHoliday'] = data['SchoolHoliday'].astype(float)
    
    features.append('StateHolidayA')
    features.append('StateHolidayB')
    features.append('StateHolidayC')
    
    data['StateHolidayA'] = 0
    data['StateHolidayB'] = 0
    data['StateHolidayC'] = 0
    
    data.loc[data['StateHoliday'] == 'a', 'StateHolidayA'] = '1'
    data.loc[data['StateHoliday'] == 'b', 'StateHolidayB'] = '1'
    data.loc[data['StateHoliday'] == 'c', 'StateHolidayC'] = '1'
    
    data['StateHolidayA'] = data['StateHolidayA'].astype(float)
    data['StateHolidayB'] = data['StateHolidayB'].astype(float)
    data['StateHolidayC'] = data['StateHolidayC'].astype(float)
    
    features.append('DayOfWeek')
    features.append('month')
    features.append('day')
    features.append('year')
    
    data['year'] = data.Date.apply(lambda x: x.split('-')[0])
    data['year'] = data['year'].astype(float)
    data['month'] = data.Date.apply(lambda x: x.split('-')[1])
    data['month'] = data['month'].astype(float)
    data['day'] = data.Date.apply(lambda x: x.split('-')[2])
    data['day'] = data['day'].astype(float)

    features.append('StoreTypeA')
    features.append('StoreTypeB')
    features.append('StoreTypeC')
    
    data['StoreTypeA'] = 0
    data['StoreTypeB'] = 0
    data['StoreTypeC'] = 0
    
    data.loc[data['StoreType'] == 'a', 'StoreTypeA'] = '1'
    data.loc[data['StoreType'] == 'b', 'StoreTypeB'] = '1'
    data.loc[data['StoreType'] == 'c', 'StoreTypeC'] = '1'
    
    data['StoreTypeA'] = data['StoreTypeA'].astype(float)
    data['StoreTypeB'] = data['StoreTypeB'].astype(float)
    data['StoreTypeC'] = data['StoreTypeC'].astype(float)

    features.append('AssortmentA')
    features.append('AssortmentB')
    
    data['AssortmentA'] = 0
    data['AssortmentB'] = 0
    
    data.loc[data['Assortment'] == 'a', 'AssortmentA'] = '1'
    data.loc[data['Assortment'] == 'b', 'AssortmentB'] = '1'
    
    data['AssortmentA'] = data['AssortmentA'].astype(float)
    data['AssortmentB'] = data['AssortmentB'].astype(float)
    
    """ State doesn't help
    features.append('StoreState')
    state_dict = get_store_state_dict()
    data['StoreState'] = 0
    
    for key in state_dict.keys():
        data.loc[data['State'] == key, 'StoreDate'] = float(state_dict[key])
    """
    
def create_model():
    print("Load the training, test and store data using pandas")
    
    train = pd.read_csv(DATA_DIR + "train.csv")
    test = pd.read_csv(DATA_DIR + "test.csv")
    store = pd.read_csv(DATA_DIR + "store.csv")
    store_states = pd.read_csv(DATA_DIR + "store_states.csv")
    
    print("Assume store open, if not provided")
    test.fillna(1, inplace=True)
    
    print("Consider only open stores for training. Closed stores wont count into the score.")
    train = train[train["Open"] != 0]
    
    print("Join with store")
    train = pd.merge(train, store, on='Store')
    train = pd.merge(train, store_states, on='Store')
    
    test = pd.merge(test, store, on='Store')
    test = pd.merge(test, store_states, on='Store')
    
    features = []
    
    print("augment features")
    new_build_features_train(features, train)
    new_build_features([], test)
    print(features)
    """
    # 1.1 - .23
    params = {"objective": "reg:linear",
              "eta": 0.3,
              "max_depth": 8,
              "subsample": 0.6,
              "colsample_bytree": 0.7,
              "silent": 1
              }
    num_trees = 50
    
    .21    
    params = {"objective": "reg:linear",
              "eta": 0.3,
              "max_depth": 7,
              "subsample": 0.9,
              "colsample_bytree": 0.7,
              "silent": 1
              }
    num_trees = 250
    # v1.2 - 0.11657
    params = {"objective": "reg:linear",
              "eta": 0.15,
              "max_depth": 8,
              "subsample": 0.7,
              "colsample_bytree": 0.7,
              "silent": 1
              }
    num_trees = 400
    
     # v1.2 - 0.11657
    params = {"objective": "reg:linear",
              "eta": 0.15,
              "max_depth": 8,
              "subsample": 0.7,
              "colsample_bytree": 0.7,
              "silent": 1
              }
    num_trees = 400
    
    # default
    params = {"objective": "reg:linear",
              "eta": 0.3,
              "max_depth": 8,
              "subsample": 0.7,
              "colsample_bytree": 0.7,
              "silent": 1
              }
    num_trees = 300
    """
     # v1.2 - 0.11657
    params = {"objective": "reg:linear",
              "eta": 0.15,
              "max_depth": 8,
              "subsample": 0.7,
              "colsample_bytree": 0.7,
              "silent": 1
              }
    num_trees = 450
    
    print("Train a XGBoost model")
    val_size = 100000
    #train = train.sort(['Date'])
    print(train.tail(1)['Date'])
    X_train, X_test = cross_validation.train_test_split(train, test_size=0.01)
    #X_train, X_test = train.head(len(train) - val_size), train.tail(val_size)
    dtrain = xgb.DMatrix(X_train[features], np.log(X_train["Sales"] + 1))
    dvalid = xgb.DMatrix(X_test[features], np.log(X_test["Sales"] + 1))
    dtest = xgb.DMatrix(test[features])
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=50, feval=rmspe_xg, verbose_eval=True)
    
    print("Validating")
    train_probs = gbm.predict(xgb.DMatrix(X_test[features]))
    indices = train_probs < 0
    train_probs[indices] = 0
    error = rmspe(np.exp(train_probs) - 1, X_test['Sales'].values)
    print('error', error)
    
    print("Make predictions on the test set")
    test_probs = gbm.predict(xgb.DMatrix(test[features]))
    indices = test_probs < 0
    test_probs[indices] = 0
    test_probs[test['Open'] == '1'] = 0
    submission = pd.DataFrame({"Id": test["Id"], "Sales": np.exp(test_probs) - 1})
    submission.to_csv(OUTPUT_DIR + "xgboost_kscript_submission.csv", index=False)
    
def split_data():
    
    train_org = pd.read_csv(DATA_DIR + "train.csv")
    #test = pd.read_csv(DATA_DIR + "test.csv")
    store = pd.read_csv(DATA_DIR + "store.csv")
    
    print("Assume store open, if not provided")
    #test.fillna(1, inplace=True)
    
    print("Consider only open stores for training. Closed stores wont count into the score.")
    train_org = train_org[train_org["Open"] != 0]
    
    print("Join with store")
    train_org = pd.merge(train_org, store, on='Store')
    #test = pd.merge(test, store, on='Store')

    import random
    bool_list = [random.random() >= .2 for _ in range(0, len(train_org))]
    
    print("augment features")
    features = []
    new_build_features_train(features, train_org)
    print(features)
    
    train = train_org[bool_list]
    test = train_org[[not item for item in bool_list]]
    
     # v1.2 - 0.11657
    params = {"objective": "reg:linear",
              "eta": 0.15,
              "max_depth": 8,
              "subsample": 0.7,
              "colsample_bytree": 0.7,
              "silent": 1
              }
    num_trees = 400
    
    print("Train a XGBoost model")
    
    print(train.tail(1)['Date'])
    X_train, X_test = cross_validation.train_test_split(train, test_size=0.01)
    #X_train, X_test = train.head(len(train) - val_size), train.tail(val_size)
    dtrain = xgb.DMatrix(X_train[features], np.log(X_train["Sales"] + 1))
    dvalid = xgb.DMatrix(X_test[features], np.log(X_test["Sales"] + 1))
    dtest = xgb.DMatrix(test[features])
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=50, feval=rmspe_xg, verbose_eval=True)
    
    print("Validating")
    train_probs = gbm.predict(xgb.DMatrix(X_test[features]))
    indices = train_probs < 0
    train_probs[indices] = 0
    error = rmspe(np.exp(train_probs) - 1, X_test['Sales'].values)
    print('error', error)
    
    print("Make predictions on the test set")
    test_probs = gbm.predict(xgb.DMatrix(test[features]))
    indices = test_probs < 0
    test_probs[indices] = 0
    error = rmspe(np.exp(test_probs) - 1, test['Sales'].values)
    print('error', error)


if __name__ == '__main__':
    create_model()
    #split_data()
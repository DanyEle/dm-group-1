import pandas as pd
from fim import apriori
"""This function bins the columns age, limit, ba, pa"""


def binCols(df):
    df['age_bin'] = pd.cut(df['age'].astype(int), 5, right=False)
    df['limit_bin'] = pd.cut(df['limit'].astype(int), 7, right=False)
    df['ba-sep_bin'] = pd.cut(df['ba-sep'].astype(int), 7, right=False)
    df['ba-aug_bin'] = pd.cut(df['ba-aug'].astype(int), 7, right=False)
    df['ba-jul_bin'] = pd.cut(df['ba-jul'].astype(int), 7, right=False)
    df['ba-jun_bin'] = pd.cut(df['ba-jun'].astype(int), 7, right=False)
    df['ba-may_bin'] = pd.cut(df['ba-may'].astype(int), 7, right=False)
    df['ba-apr_bin'] = pd.cut(df['ba-apr'].astype(int), 7, right=False)
    df['pa-sep_bin'] = pd.cut(df['pa-sep'].astype(int), 5, right=False)
    df['pa-aug_bin'] = pd.cut(df['pa-aug'].astype(int), 5, right=False)
    df['pa-jul_bin'] = pd.cut(df['pa-jul'].astype(int), 5, right=False)
    df['pa-jun_bin'] = pd.cut(df['pa-jun'].astype(int), 5, right=False)
    df['pa-may_bin'] = pd.cut(df['pa-may'].astype(int), 5, right=False)
    df['pa-apr_bin'] = pd.cut(df['pa-apr'].astype(int), 5, right=False)
    #dropping all the original columns
    cols2drop = [
        'age', 'limit', 'ba-sep', 'ba-aug', 'ba-jul', 'ba-jun', 'ba-may',
        'ba-apr', 'pa-sep', 'pa-aug', 'pa-jul', 'pa-jun', 'pa-may', 'pa-apr'
    ]
    df.drop(cols2drop, axis=1, inplace=True)


"""This function adds a label to all numerical values, in order to recognize their origin in pattern mining"""


def remapCols(df):
    df['age_bin'] = df['age_bin'].astype(str) + '_age'
    df['limit_bin'] = df['limit_bin'].astype(str) + '_limit'

    df['ba-sep_bin'] = df['ba-sep_bin'].astype(str) + '_ba-sep'
    df['ba-aug_bin'] = df['ba-aug_bin'].astype(str) + '_ba-aug'
    df['ba-jul_bin'] = df['ba-jul_bin'].astype(str) + '_ba-jul'
    df['ba-jun_bin'] = df['ba-jun_bin'].astype(str) + '_ba-jun'
    df['ba-may_bin'] = df['ba-may_bin'].astype(str) + '_ba-may'
    df['ba-apr_bin'] = df['ba-apr_bin'].astype(str) + '_ba-apr'

    df['pa-sep_bin'] = df['pa-sep_bin'].astype(str) + '_pa-sep'
    df['pa-aug_bin'] = df['pa-aug_bin'].astype(str) + '_pa-aug'
    df['pa-jul_bin'] = df['pa-jul_bin'].astype(str) + '_pa-jul'
    df['pa-jun_bin'] = df['pa-jun_bin'].astype(str) + '_pa-jun'
    df['pa-may_bin'] = df['pa-may_bin'].astype(str) + '_pa-may'
    df['pa-apr_bin'] = df['pa-apr_bin'].astype(str) + '_pa-apr'

    df['ps-sep'] = df['ps-sep'].astype(str) + '_ps-sep'
    df['ps-aug'] = df['ps-aug'].astype(str) + '_ps-aug'
    df['ps-jul'] = df['ps-jul'].astype(str) + '_ps-jul'
    df['ps-jun'] = df['ps-jun'].astype(str) + '_ps-jun'
    df['ps-may'] = df['ps-may'].astype(str) + '_ps-may'
    df['ps-apr'] = df['ps-apr'].astype(str) + '_ps-apr'


def sortCols(df):
    columnsTitles = [
        "credit_default", "sex", "education", "status", "age_bin", "limit_bin",
        "ps-sep", "ps-aug", "ps-jul", "ps-jun", "ps-may", "ps-apr",
        "pa-sep_bin", "pa-aug_bin", "pa-jul_bin", "pa-jun_bin", "pa-may_bin",
        "pa-apr_bin", "ba-sep_bin", "ba-aug_bin", "ba-jul_bin", "ba-jun_bin",
        "ba-may_bin", "ba-apr_bin"
    ]
    df = df[columnsTitles]


def allFreqPatterns(baskets, confidence):
    #zmin -> minimum number of items per itemset
    #supp -> support
    #conf -> confidence
    #target -> a = all
    #report -> ascl =
    #(a, absolute itemset support);
    #(s, relative itemset support as a fraction);
    #(c, rule confidence as a fraction);
    #(l, lift value of a rule)
    d = ({})  #dictionary to store rules
    for s in range(1, 100):
        rules = apriori(
            baskets, supp=s, zmin=2, target='a', conf=confidence, report='scl')
        d[s] = ({rules})
    pickle.dump(d, open("allFreqPatterns.p", "wb"))


def closedFreqPatterns(baskets, confidence):
    d = ({})  #dictionary to store rules
    for s in range(1, 100):
        rules = apriori(
            baskets, supp=s, zmin=2, target='c', conf=confidence, report='scl')
        d[s] = ({rules})
    pickle.dump(d, open("closedFreqPatterns.p", "wb"))


def minimalFreqPatterns(baskets, confidence):
    d = ({})  #dictionary to store rules
    for s in range(1, 100):
        rules = apriori(
            baskets, supp=s, zmin=2, target='m', conf=confidence, report='scl')
        d[s] = ({rules})
    pickle.dump(d, open("minimalFreqPatterns.p", "wb"))


def associationRules(baskets):
    d = ({})  #dictionary to store rules
    for s in range(1, 100):
        for c in range(1, 100, 10):
            rules = apriori(
                baskets, supp=s, zmin=2, target='r', conf=c, report='scl')
            d[(s, c)] = ({rules})
    pickle.dump(d, open("associationRules.p", "wb"))


def dataVisual(confidence):
    with open('allFreqPatterns.p', 'rb') as f:
        d = pickle.load(f)
        for k, v in d.items():
            numRules = len(v)
            if (numRules == 0):
                break
            print('Supp: ', k, ' Conf: ', confidence, ' Number of rules:',
                  numRules)
            for r in v:
                if r[0] == 'yes':
                    print(r)
    with open('closedFreqPatterns.p', 'rb') as f:
        d = pickle.load(f)
        for k, v in d.items():
            numRules = len(v)
            if (numRules == 0):
                break
            print('Supp: ', k, ' Conf: ', confidence, ' Number of rules:',
                  numRules)
            for r in v:
                if r[0] == 'yes':
                    print(r)
    with open('minimalFreqPatterns.p', 'rb') as f:
        d = pickle.load(f)
        for k, v in d.items():
            numRules = len(v)
            if (numRules == 0):
                break
            print('Supp: ', k, ' Conf: ', confidence, ' Number of rules:',
                  numRules)
            for r in v:
                if r[0] == 'yes':
                    print(r)
    with open('associationRules.p', 'rb') as f:
        d = pickle.load(f)
        for k, v in d.items():
            numRules = len(v)
            if (numRules == 0):
                break
            print('Supp: ', k[0], ' Conf: ', k[1], ' Number of rules:',
                  numRules)
            numYes = 0
            for r in v:
                if r[0] == 'yes':
                    #print(r)
                    numYes = numYes + 1
            set(v[0][1])

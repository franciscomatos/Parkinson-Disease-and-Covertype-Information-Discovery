import warnings
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules #for ARM
from sklearn.feature_selection import SelectKBest, f_classif
import aux.functions as func


def patternMining(df, dataClass, trnX, trnY):

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # SelectKBest returns a numpy array and we want a dataframe
        X_new = SelectKBest(f_classif, k=10).fit(trnX, trnY)
        cols = X_new.get_support(indices=True)
        bestdf = df.iloc[:,cols]

    newdf = bestdf.copy()
    dummified_df = None
    # pd dataset needs discretization, covtype doesn't
    if dataClass == "class":
        # discretize real-valued attributes
        for col in newdf:
            if col not in [dataClass]:
                newdf[col] = pd.qcut(newdf[col],3,labels=['0','1','2'])
        newdf.head(5)

        # dummify the new discretized values
        dummylist = []
        for att in newdf:
           dummylist.append(pd.get_dummies(newdf[[att]]))
        dummified_df = pd.concat(dummylist, axis=1)
        dummified_df.head(5)
    else:
        dummified_df = func.dummify(newdf, newdf.columns)



    # iteratively decreasing support
    print("a) Iteratively decreasing support results:")
    frequent_itemsets = {}
    minpaterns = 30
    minsup = 1.0
    while minsup > 0:
        minsup = minsup * 0.9
        frequent_itemsets = apriori(dummified_df, min_support=0.2, use_colnames=True)
        if len(frequent_itemsets) >= minpaterns:
            print("Minimum support:", minsup)
            break
    print("Number of patterns found:", len(frequent_itemsets))
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

    # association rules with different confidence thresholds
    print("b) Association rules results:")
    confidence = [0.7, 0.9]
    for c in confidence:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=c)
        rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
        print("For confidence =", c, ":")

        print("Rule with > 2 items with the biggest confidence: ")
        max = rules.loc[rules[(rules['antecedent_len'] > 2)]['confidence'].idxmax()]
        for i in range(len(max)):
            print(str(rules.columns[i]) + ":", max[i], end=' | ')
        print()

        print("Rule with > 2 items with the smallest confidence: ")
        min = rules.loc[rules[(rules['antecedent_len'] > 2)]['confidence'].idxmin()]
        for i in range(len(min)):
            print(str(rules.columns[i]) + ":", min[i], end=' | ')

        print()


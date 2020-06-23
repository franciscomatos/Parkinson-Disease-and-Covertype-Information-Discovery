import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import unsupervised.clustering as cl
import unsupervised.patternMining as pm



PD = pd.DataFrame()
CT = pd.DataFrame()


# right now the preprocessing only removes the class and performs normalization
def preprocessing(data, dataClass):
    
    noClassData = data.loc[:, data.columns != dataClass]
    transfData = MinMaxScaler().fit(noClassData)
    return pd.DataFrame(transfData.transform(noClassData), columns=noClassData.columns)


def associationRules(data, dataClass, trnX, trnY):
    print("2.1 Association Rules")
    pm.patternMining(data, dataClass, trnX, trnY)


def clustering(trnX, trnY):

    print("2.2 Clustering")
    # k-means with different k-values
    cl.kmeans(trnX)

    # agglomerative clustering with different critteria
    cl.agglomerative(trnX, trnY)


def unsupervised(data, processedData, dataClass):
    print("2. Unsupervised Learning:")

    # first we need to split the data into training and testing set in order to avoid overfitting
    X = processedData.values
    y = data.loc[:, dataClass].values

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    # association rules
    associationRules(dataframe, dataClass, trnX, trnY)

    # clustering
    clustering(trnX, trnY)


def report(source, dataframe, task):

    dataClass = None
    if source == "PD":
        dataClass = 'class'
    if source == "CT":
        dataClass = 'Cover_Type'


    if task == "preprocessing":
        # preprocessing tasks
        processedData = preprocessing(dataframe, dataClass)
    elif task == "unsupervised":
        # preprocessing task
        processedData = preprocessing(dataframe, dataClass)
        unsupervised(dataframe, processedData, dataClass)
    elif task == "classification":
        # preprocessing task

        # classifiers
            ## nb
                ### sugested parameterization
                ### confusion matrix
            ## instance-based
                ### sugested parameterization
                ### confusion matrix
            ## dt
                ### sugested parameterization
                ### confusion matrix
            ## rf
                ### sugested parameterization
                ### confusion matrix
            ## xgboost
                ### sugested parameterization
                ### confusion matrix

        # comparative performace
        print("gonna do classification")
    else:
        print(task)


def readData(source):

    pdData = None
    covData = None

    # original datasets
    if source == "PD":
        # parkinson disease
        pdData = pd.read_csv('../../Data/pd_speech_features.csv', index_col='id', sep=',', header=1, decimal='.',
                             parse_dates=True, infer_datetime_format=True)
    else:
        # forest covertype data
        # .data file
        # first we need the col names
        cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                'Horizontal_Distance_To_Fire_Points']

        # BINARY DATA
        # 4 types of wilderness areas + 40 types of soil
        for i in range(0, 44):
            name = 'y' + str(i)
            cols += [name, ]

        # finally the class
        cols += ['Cover_Type', ]

        # we read the data from the file
        with open("../../Data/covtype.data", "r") as file:
            line = file.readline().strip()
            covTable = [line.split(',')]
            while line:
                line = file.readline().strip()
                # avoid adding the last non existing line to the list
                if line:
                    covTable += [line.split(','), ]

        covData = pd.DataFrame(covTable, columns=cols)

    return pdData, covData


if __name__ == '__main__':

    '''A: read arguments'''
    args = sys.stdin.readline().rstrip('\n').split(' ')
    n, source, task = int(args[0]), args[1], args[2]
    
    '''B: read dataset'''
    data, header = [], sys.stdin.readline().strip().split(',')
    if source.strip() == 'PD':
        for i in range(n-1):
            data.append(list(map(float, sys.stdin.readline().strip().split(','))))
    else:
        for i in range(n-1):
            data.append(list(map(int, sys.stdin.readline().strip().split(','))))

    dataframe = pd.DataFrame(data, columns=header)

    if source.strip() == 'PD':
        dataframe.set_index('id', inplace=True, drop=True)

    # PD, CT = readData(source.strip())

    '''C: output results'''
    report(source.strip(), dataframe, task.strip())

from sklearn.ensemble.forest import ForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from numpy import random
from copy import deepcopy
import math


# TimeSeriesForest: Port from Java
# Implementation of Deng 's Time Series Forest
# Time Series Forest(TimeSeriesForest) Deng 2013:
# @article
# {deng13forest,
#  author = {H.Deng and G.Runger and E.Tuv and M.Vladimir},
#           title = {A time series forest for classification and feature extraction},
# journal = {Information Sciences},
# volume = {239},
# year = {2013}
#
# Overview: Input n series length m
# for each tree
#     sample sqrt(m) intervals
#     build tree on these features
# ensemble the trees with majority vote
#
# Three interval features: mean, standard deviation and slope.
#
# This implementation may deviate from the original, as it is using the
# same structure as random forest.
# fit:
# 1. Pick sqrt(m) intervals
# 2. Construct instances of three features
# 3. build a sqrt(m) RandomTree classifiers
#
# predict:
# majority vote RandomTree classifiers
#
# Splitting criteria has a tiny refinement.Ties in entropy gain
# are split with a further stat called margin that measures the distance of the split point
# to the closest data.So if the split value for feature f=f_1, ...f_n is v the margin is defined
# as
#
# margin= min{| f_i-v |}
#


class TimeSeriesForest(ForestClassifier):

    def __init__(self,
                 n_trees=200,
                 random_state=None,
                 verbose=0):
        super(TimeSeriesForest, self).__init__(
            base_estimator=DecisionTreeClassifier(criterion="entropy"),
            n_estimators=n_trees)
        self._num_trees=n_trees
        self.verbose = verbose
        self.random_state = random_state
        random.seed(random_state)
        self._num_classes = 0
        self._num_atts = 0
        self._num_intervals = 0
        self._classifiers = []
        self._intervals=[]
        self._min_interval=3
        self.dim_to_use = 0

    def fit(self, X, y, sample_weight=None):

        if isinstance(X, pd.DataFrame):
            if isinstance(X.iloc[0,self.dim_to_use],pd.Series):
                X = np.asarray([a.values for a in X.iloc[:,0]])
            else:
                raise TypeError("Input should either be a 2d numpy array, or a pandas dataframe containing Series objects")
        n_samps, self._num_atts = X.shape

        self._num_classes = np.unique(y).shape[0]

        self.classes_ = list(set(y))
        self.classes_.sort()
        self._num_intervals = int(math.sqrt(self._num_atts))
        if self._num_intervals==0:
            self._num_intervals=1
        if self._num_atts <self._min_interval:
            self._min_interval=self._num_atts
        self._intervals=np.zeros((self._num_trees,3*self._num_intervals,2),dtype=int)

        for i in range(0, self._num_trees):
            transformed_x = np.empty(shape=(3*self._num_intervals,n_samps))
            for j in range(0,self._num_intervals):
                self._intervals[i][j][0]=random.randint(self._num_atts-self._min_interval)
                length=random.randint(self._num_atts-self._intervals[i][j][0]-1)
                if length < self._min_interval:
                    length=self._min_interval
                self._intervals[i][j][1]=self._intervals[i][j][0]+length
#Transforms here, currently just hard coding it
                means = np.mean(X[:, self._intervals[i][j][0]:self._intervals[i][j][1]], axis=1)
                std_dev = np.std(X[:, self._intervals[i][j][0]:self._intervals[i][j][1]], axis=1)
                slope = self.lsq_fit(X[:, self._intervals[i][j][0]:self._intervals[i][j][1]])
                transformed_x[3*j]=means
                transformed_x[3*j+1]=std_dev
                transformed_x[3*j+2]=slope
            tree = deepcopy(self.base_estimator)
            transformed_x=transformed_x.T
            tree.fit(transformed_x, y)
            self._classifiers.append(tree)

    def predict(self, X):
        proba=self.predict_proba(X)
        return [self.classes_[np.argmax(prob)] for prob in proba]

    def predict_proba(self, X):

        if isinstance(X, pd.DataFrame):
            if isinstance(X.iloc[0,self.dim_to_use],pd.Series):
                X = np.asarray([a.values for a in X.iloc[:,0]])
            else:
                raise TypeError("Input should either be a 2d numpy array, or a pandas dataframe containing Series objects")

        #HERE Do transform againnum_att
        n_samps, num_atts = X.shape
        if num_atts != self._num_atts:
            print(" ERROR throw an exception or somesuch")
            return
        sums = np.zeros((X.shape[0],self._num_classes), dtype=np.float64)
        for i in range(0, self._num_trees):
            transformed_x = np.empty(shape=(3*self._num_intervals,n_samps),dtype=np.float32)
            for j in range(0,self._num_intervals):
                means = np.mean(X[:, self._intervals[i][j][0]:self._intervals[i][j][1]], axis=1)
                std_dev = np.std(X[:, self._intervals[i][j][0]:self._intervals[i][j][1]], axis=1)
                slope = self.lsq_fit(X[:, self._intervals[i][j][0]:self._intervals[i][j][1]])
                transformed_x[3*j]=means
                transformed_x[3*j+1]=std_dev
                transformed_x[3*j+2]=slope
            transformed_x=transformed_x.T
            sums += self._classifiers[i].predict_proba(transformed_x)

        output = sums / (np.ones(self._num_classes) * self.n_estimators)
        return output

    def lsq_fit(self, Y):
        x = np.arange(Y.shape[1]) + 1
        slope = (np.mean(x * Y, axis=1) - np.mean(x) * np.mean(Y, axis=1)) / ((x * x).mean() - x.mean() ** 2)
        return slope


def load_from_tsfile_to_dataframe(file_path, file_name, replace_missing_vals_with='NaN'):
    data_started = False
    instance_list = []
    class_val_list = []

    has_time_stamps = False
    has_class_labels = False

    uses_tuples = False

    is_first_case = True
    with open(file_path + file_name, 'r') as f:
        for line in f:

            if line.strip():
                if "@timestamps" in line.lower():
                    if "true" in line.lower():
                        has_time_stamps = True
                        raise Exception("Not suppoorted yet")  # we don't have any data formatted to test with yet
                    elif "false" in line.lower():
                        has_time_stamps = False
                    else:
                        raise Exception("invalid timestamp argument")

                if "@classlabel" in line.lower():
                    if "true" in line:
                        has_class_labels = True
                    elif "false" in line:
                        has_class_labels = False
                    else:
                        raise Exception("invalid classLabel argument")

                if "@data" in line.lower():
                    data_started = True
                    continue

                # if the 'data tag has been found, the header information has been cleared and now data can be loaded
                if data_started:
                    line = line.replace("?", replace_missing_vals_with)
                    dimensions = line.split(":")

                    # perhaps not the best way to do this, but on the first row, initialise stored depending on the
                    # number of dimensions that are present and determine whether data is stored in a list or tuples
                    if is_first_case:
                        num_dimensions = len(dimensions)
                        if has_class_labels:
                            num_dimensions -= 1
                        is_first_case = False
                        for dim in range(0, num_dimensions):
                            instance_list.append([])
                        if dimensions[0].startswith("("):
                            uses_tuples = True

                    this_num_dimensions = len(dimensions)
                    if has_class_labels:
                        this_num_dimensions -= 1

                    # assuming all dimensions are included for all series, even if they are empty. If this is not true
                    # it could lead to confusing dimension indices (e.g. if a case only has dimensions 0 and 2 in the
                    # file, dimension 1 should be represented, even if empty, to make sure 2 doesn't get labelled as 1)
                    if this_num_dimensions != num_dimensions:
                        raise Exception("inconsistent number of dimensions")

                    # go through each dimension that is represented in the file
                    for dim in range(0, num_dimensions):

                        # handle whether tuples or list here
                        if uses_tuples:
                            without_brackets = dimensions[dim].replace("(", "").replace(")", "").split(",")
                            without_brackets = [float(i) for i in without_brackets]

                            indices = []
                            data = []
                            i = 0
                            while i < len(without_brackets):
                                indices.append(int(without_brackets[i]))
                                data.append(without_brackets[i + 1])
                                i += 2

                            instance_list[dim].append(pd.Series(data, indices))
                        else:
                            # if the data is expressed in list form, just read into a pandas.Series
                            data_series = dimensions[dim].split(",")
                            data_series = [float(i) for i in data_series]
                            instance_list[dim].append(pd.Series(data_series))

                    if has_class_labels:
                        class_val_list.append(dimensions[num_dimensions].strip())

    # note: creating a pandas.DataFrame here, NOT an xpandas.xdataframe
    x_data = pd.DataFrame(dtype=np.float32)
    for dim in range(0, num_dimensions):
        x_data['dim_' + str(dim)] = instance_list[dim]

    if has_class_labels:
        return x_data, np.asarray(class_val_list)
    #
    # # otherwise just return an XDataFrame
    return x_data


if __name__ == "__main__":

    dataset = "Gunpoint"
    train_x, train_y = load_from_tsfile_to_dataframe(file_path="C:/temp/sktime_temp_data/" + dataset + "/", file_name=dataset + "_TRAIN.ts")

    print(train_x.iloc[0:10])

    tsf = TimeSeriesForest()
    tsf.fit(train_x.iloc[0:10], train_y[0:10])
    preds = tsf.predict(train_x.iloc[10:20])
    print(preds)

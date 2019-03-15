import LoadData as ld
import os
import numpy as np
import time
import sys
from classifiers.dictionary_based import BOSS
from classifiers.time_domain_classification import RotationForest as rf
import classifiers.interval_based.TimeSeriesForest as tsf


def time_series_slope(y):
    n = y.shape[0]
    if n < 2:
        return 0
    else:
        x = np.arange(n) + 1
        x_mu = x.mean()
        return (((x * y).mean() - x_mu * y.mean())
                / ((x ** 2).mean() - x_mu ** 2))

def oneProblemTest(dataset_name):
    problem_path = "E:/TSCProblems/"
    results_path="E:/Results/Python/"
    suffix = "_TRAIN.arff"
    train_x, train_y = ld.load_csv(problem_path + "/"+dataset_name + "/"+dataset_name+ suffix)
    rotF=rf.RotationForest(n_estimators=200)
    rotF.fit(train_x,train_y)
    suffix = "_TEST.arff"
    test_x, test_y = ld.load_csv(problem_path + "/"+dataset_name +"/"+dataset_name+suffix)
    pred_y=rotF.predict(test_x)
    correct =0
    for i in range(0,pred_y.__len__()):
        if pred_y[i] == test_y[i]:
            correct+=1
    print(correct)
    ac=correct/pred_y.__len__()
    print(ac)


def defaultTrainTestFold(classifier, dataset_name, results_path, problem_path):
    current_milli_time = lambda: int(round(time.time() * 1000))
    # Write results to file
    results_path=results_path+ dataset_name
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(results_path+"/testFold0.csv"):
        suffix = "_TRAIN.arff"
        full_path = problem_path + "/" + dataset_name + "/"
        train_x, train_y = ld.load_to_pandas(problem_path + "/" + dataset_name + "/" + dataset_name + suffix)
        buildTime = current_milli_time()
        classifier.fit(train_x, train_y)
        buildTime=current_milli_time()-buildTime
        suffix = "_TEST.arff"

        test_x, test_y = ld.load_to_pandas(problem_path + "/" + dataset_name + "/" + dataset_name + suffix)
        testTime= current_milli_time()
        pred_y = classifier.predict(test_x)
        testTime=current_milli_time()-testTime
#Map the classes on to range 0 ... numClasses-1
        num_classes = np.unique(train_y).shape[0]
        class_dictionary={}
        for index, selected_class in enumerate(np.unique(train_y)):
            class_dictionary[selected_class]=index

        for temp in class_dictionary:
            print(str(temp))
        predA_y = classifier.predict_proba(test_x)

        correct = 0
        for i in range(0, pred_y.__len__()):
            if pred_y[i] == test_y[i]:
                correct += 1
        print(correct)
        ac = correct / pred_y.__len__()
        print(" Accuracy =")
        print(ac)
        print("Writing results .....")
        with open(results_path+"/testFold0.csv","w") as f:
            f.write("RotF.py,")
            f.write(results_path)
            f.write(",test\nBuildTime,")
            f.write(str(buildTime))
            f.write(",TestTime,")
            f.write(str(testTime))
            f.write(",NumberOfTrees,")
#            f.write(str(classifier.n_estimators))
#            f.write(",NumberOfIntervals,")
#            f.write(str(classifier._num_intervals))
            f.write("\n")
            f.write(str(ac))
            f.write("\n")
            for i in range(0,pred_y.__len__()):
                f.write(str(class_dictionary[test_y[i]])+",")
                f.write(str(class_dictionary[pred_y[i]]))
                f.write(",")
                for j in range(0,predA_y[i].__len__()):
                    f.write(",")
                    f.write(str(predA_y[i][j]))
                f.write("\n")
    else:
        print(dataset_name+" testFold0.csv already exists")



if __name__ == "__main__":

    if len(sys.argv) >= 5:
        cls = sys.argv[1]
        problem = sys.argv[2]
        dataDir = sys.argv[3]
        resultsDir = sys.argv[4]
    else:
        cls = "BOSSPY"
        problem = "ItalyPowerDemand"
        dataDir = "Z:/Data/TSCProblems2018/"
        resultsDir = "Z:/Results/Python/"

    print(problem)
    if cls == "RotF":
        classifier = rf.RotationForest(n_estimators=200)
    elif cls == "TSF_Tony":
        classifier = tsf.TimeSeriesForest(n_trees=500)
    elif cls == "BOSSPY":
        classifier = BOSS.BOSSClassifier()
    else:
        print("Invalid Classifier: " + cls)
        exit()

    defaultTrainTestFold(classifier, problem, resultsDir + cls + "/Predictions/", dataDir)

from transformers.BOSSTransformer import BOSSTransform
import numpy as np
import random
import sys
import pandas as pd

class BOSSClassifier():

    def __init__(self):
        self.useRandomSampling = False
        self.ensembleSize = 100
        self.seed = 0

        self.correctThreshold = 0.92
        self.maxEnsembleSize = 500

        self.wordLengths = [16, 14, 12, 10, 8]
        self.normOptions = [True, False]
        self.alphabetSize = 4

        self.classifiers = []
        self._num_classes = 0
        self.classes_ = []
        self.class_dictionary = {}
        self.numClassifiers = 0
        self.dim_to_use = 0 #For the multivariate case treating this as a univariate classifier


    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            if isinstance(X.iloc[0,self.dim_to_use],pd.Series):
                X = np.asarray([a.values for a in X.iloc[:,0]])
            else:
                raise TypeError("Input should either be a 2d numpy array, or a pandas dataframe containing Series objects")


        num_insts, num_atts = X.shape
        self._num_classes = np.unique(y).shape[0]
        self.classes_ = list(set(y))
        for index, classVal in enumerate(self.classes_):
            self.class_dictionary[classVal] = index

        minWindow = 10
        maxWindowSearches = num_atts/4
        winInc = (int)((num_atts - minWindow) / maxWindowSearches)
        if winInc < 1: winInc = 1

        if self.useRandomSampling:
            random.seed(self.seed)

            while len(self.classifiers) < self.ensembleSize:
                wordLen = self.wordLengths[random.randint(0, len(self.wordLengths)-1)]
                winSize = minWindow + winInc * random.randint(0, maxWindowSearches)
                if winSize > maxWindowSearches: winSize = maxWindowSearches
                normalise = random.random() > 0.5

                boss = BOSSIndividual(winSize, self.wordLengths[wordLen], self.alphabetSize, normalise)
                boss.fit(X, y)
                boss.clean()
                self.classifiers.append(boss)
        else:
            maxAcc = -1
            minMaxAcc = -1

            for i, normalise in enumerate(self.normOptions):
                for winSize in range(minWindow, num_atts+1, winInc):
                    boss = BOSSIndividual(winSize, self.wordLengths[0], self.alphabetSize, normalise)
                    boss.fit(X, y)

                    bestAccForWinSize = -1

                    for n, wordLen in enumerate(self.wordLengths):
                        if n > 0:
                            boss = boss.shortenBags(wordLen)

                        correct = 0
                        for g in range(num_insts):
                            c = boss.train_predict(g)
                            if (c == y[g]):
                                correct += 1

                        accuracy = correct/num_insts
                        if (accuracy >= bestAccForWinSize):
                            bestAccForWinSize = accuracy
                            bestClassifierForWinSize = boss
                            bestWordLen = wordLen

                    if self.include_in_ensemble(bestAccForWinSize, maxAcc, minMaxAcc, len(self.classifiers)):
                        bestClassifierForWinSize.clean()
                        bestClassifierForWinSize.setWordLen(bestWordLen)
                        bestClassifierForWinSize.accuracy = bestAccForWinSize
                        self.classifiers.append(bestClassifierForWinSize)

                        if bestAccForWinSize > maxAcc:
                            maxAcc = bestAccForWinSize

                            for c, classifier in enumerate(self.classifiers):
                                if classifier.accuracy < maxAcc * self.correctThreshold:
                                    self.classifiers.remove(classifier)

                        minMaxAcc, minAccInd = self.worst_of_best()

                        if len(self.classifiers) > self.maxEnsembleSize:
                            del self.classifiers[minAccInd]
                            minMaxAcc, minAccInd = self.worst_of_best()

        self.numClassifiers = len(self.classifiers)

        #train estimate stuff
        f = 1

    def predict(self, X):
        return [self.classes_[np.argmax(prob)] for prob in self.predict_proba(X)]

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            if isinstance(X.iloc[0,self.dim_to_use],pd.Series):
                X = np.asarray([a.values for a in X.iloc[:,0]])
            else:
                raise TypeError("Input should either be a 2d numpy array, or a pandas dataframe containing Series objects")

        sums = np.zeros((X.shape[0], self._num_classes))

        for i, clf in enumerate(self.classifiers):
            preds = clf.predict(X)
            for n, val in enumerate(preds):
                sums[n,self.class_dictionary.get(val, -1)] += 1

        dists = sums / (np.ones(self._num_classes) * self.numClassifiers)
        return dists

    def include_in_ensemble(self, acc, maxAcc, minMaxAcc, size):
        if acc >= maxAcc * self.correctThreshold:
            if size >= self.maxEnsembleSize:
                return acc > minMaxAcc
            else:
                return True
        return False

    def worst_of_best(self):
        minAcc = -1;
        minAccInd = 0

        for c, classifier in enumerate(self.classifiers):
            if classifier.accuracy < minAcc:
                minAcc = classifier.accuracy
                minAccInd = c

        return minAcc, minAccInd

    def findEnsembleTrainAcc(self, X, y):
        num_inst = X.shape[0]
        results = np.zeros((2 + self._num_classes, num_inst + 1))
        correct = 0

        for i in range(num_inst):
            sums = np.zeros(self._num_classes)

            for n in range(len(self.classifiers)):
                sums[self.class_dictionary.get(self.classifiers[n].train_predict(i), -1)] += 1

            dists = sums / (np.ones(self._num_classes) * self.numClassifiers)
            c = dists.argmax()

            if c == self.class_dictionary.get(y[i], -1):
                correct += 1

            results[0][i+1] = self.class_dictionary.get(y[i], -1)
            results[1][i+1] = c

            for n in range(self._num_classes):
                results[2+n][i+1] = dists[n]

        results[0][0] = correct/num_inst
        return results

class BOSSIndividual:

    def __init__(self, windowSize, wordLength, alphabetSize, norm):
        self.windowSize = windowSize
        self.wordLength = wordLength
        self.alphabetSize = alphabetSize
        self.norm = norm

        self.transform = BOSSTransform(windowSize, wordLength, alphabetSize, norm)
        self.transformedData = []
        self.classVals = []
        self.accuracy = 0

    def fit(self, X, y):
        self.transformedData = self.transform.fit(X)
        self.classVals = y

    def predict(self, X):
        num_insts, num_atts = X.shape
        classes = np.zeros(num_insts, dtype=np.int_)

        for i in range(num_insts):
            testBag = self.transform.transform_single(X[i, :])
            bestDist = sys.float_info.max
            nn = -1

            for n, bag in enumerate(self.transformedData):
                dist = self.BOSSDistance(testBag, bag, bestDist)

                if dist < bestDist:
                    bestDist = dist;
                    nn = self.classVals[n]

            classes[i] = nn

        return classes

    def train_predict(self, train_num):
        testBag = self.transformedData[train_num]
        bestDist = sys.float_info.max
        nn = -1

        for n, bag in enumerate(self.transformedData):
            if n == train_num:
                continue

            dist = self.BOSSDistance(testBag, bag, bestDist)

            if dist < bestDist:
                bestDist = dist;
                nn = self.classVals[n]

        return nn

    def BOSSDistance(self, bagA, bagB, bestDist):
        dist = 0

        for word, valA in bagA.items():
            valB = bagB.get(word, 0)
            dist += (valA-valB)*(valA-valB)

            if dist > bestDist:
                return sys.float_info.max

        return dist

    def shortenBags(self, wordLen):
        newBOSS = BOSSIndividual(self.windowSize, wordLen, self.alphabetSize, self.norm)
        newBOSS.transform = self.transform
        newBOSS.transformedData = self.transform.shorten_bags(wordLen)
        newBOSS.classVals = self.classVals

        return newBOSS

    def clean(self):
        self.transform.words = None

    def setWordLen(self, wordLen):
        self.wordLength = wordLen
        self.transform.wordLength = wordLen

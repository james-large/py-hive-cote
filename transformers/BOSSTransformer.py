import numpy as np
import math
import sys
from transformers.Transformer import Transformer


class BOSSTransform(Transformer):

    def __init__(self, windowSize, wordLength, alphabetSize, norm):
        self.words = []
        self.breakpoints = []

        self.inverseSqrtWindowSize = 1 / math.sqrt(windowSize)
        self.windowSize = windowSize
        self.wordLength = wordLength
        self.alphabetSize = alphabetSize
        self.norm = norm

        self.num_insts = 0
        self.num_atts = 0

    def fit(self, X):
        self.num_insts, self.num_atts = X.shape
        self.breakpoints = self.MCB(X)

        bags = []

        for i in range(self.num_insts):
            dfts = self.MFT(X[i, :])
            bag = {}
            lastWord = -1

            words = []

            for window in range(dfts.shape[0]):
                word = self.createWord(dfts[window])
                words.append(word)
                lastWord = self.addToBag(bag, word, lastWord)

            self.words.append(words)
            bags.append(bag)

        return bags

    def transform_single(self, series):
        dfts = self.MFT(series)
        bag = {}
        lastWord = -1

        for window in range(dfts.shape[0]):
            word = self.createWord(dfts[window])
            lastWord = self.addToBag(bag, word, lastWord)

        return bag

    def MCB(self, X):
        numWindowsPerInst = math.ceil(self.num_atts / self.windowSize)
        dft = np.zeros((self.num_insts, numWindowsPerInst, int((self.wordLength / 2)*2)))

        for i in range(X.shape[0]):
            split = np.split(X[i, :], np.linspace(self.windowSize, self.windowSize*(numWindowsPerInst-1),
                                              numWindowsPerInst-1, dtype=np.int_))
            split[-1] = X[i, self.num_atts - self.windowSize:self.num_atts]

            for n, row in enumerate(split):
                dft[i, n] = self.DFT(row)

        totalNumWindows = self.num_insts * numWindowsPerInst
        breakpoints = np.zeros((self.wordLength, self.alphabetSize))

        for letter in range(self.wordLength):
            column = np.zeros(totalNumWindows)

            for inst in range(self.num_insts):
                for window in range(numWindowsPerInst):
                    column[(inst * numWindowsPerInst) + window] = round(dft[inst][window][letter] * 100) / 100

            column = np.sort(column)

            binIndex = 0
            targetBinDepth = totalNumWindows / self.alphabetSize

            for bp in range(self.alphabetSize - 1):
                binIndex += targetBinDepth
                breakpoints[letter][bp] = column[int(binIndex)]

            breakpoints[letter][self.alphabetSize - 1] = sys.float_info.max

        return breakpoints

    def DFT(self, series):
        length = len(series)
        outputLength = int(self.wordLength / 2)
        start = 1 if self.norm else 0

        std = np.std(series)
        if std == 0: std = 1
        normalisingFactor = self.inverseSqrtWindowSize / std

        dft = np.zeros(outputLength * 2)

        for i in range(start, start + outputLength):
            idx = (i - start) * 2

            for n in range(length):
                dft[idx] += series[n] * math.cos(2 * math.pi * n * i / length)
                dft[idx + 1] += -series[n] * math.sin(2 * math.pi * n * i / length)

        dft *= normalisingFactor

        return dft

    def DFTunnormed(self, series):
        length = len(series)
        outputLength = int(self.wordLength / 2)
        start = 1 if self.norm else 0

        dft = np.zeros(outputLength * 2)

        for i in range(start, start + outputLength):
            idx = (i - start) * 2

            for n in range(length):
                dft[idx] += series[n] * math.cos(2 * math.pi * n * i / length)
                dft[idx + 1] += -series[n] * math.sin(2 * math.pi * n * i / length)

        return dft

    def MFT(self, series):
        startOffset = 2 if self.norm else 0
        l = self.wordLength + self.wordLength % 2
        phis = np.zeros(l)

        for i in range(0, l, 2):
            half = -(i + startOffset)/2
            phis[i] = math.cos(2 * math.pi * half / self.windowSize);
            phis[i+1] = -math.sin(2 * math.pi * half / self.windowSize)

        end = max(1, len(series) - self.windowSize + 1)
        stds = self.calcIncrementalMeanStd(series, end)
        transformed = np.zeros((end, l))
        mftData = None

        for i in range(end):
            if i > 0:
                for n in range(0, l, 2):
                    real1 = mftData[n] + series[i + self.windowSize - 1] - series[i - 1]
                    imag1 = mftData[n + 1]
                    real = real1 * phis[n] - imag1 * phis[n + 1]
                    imag = real1 * phis[n + 1] + phis[n] * imag1
                    mftData[n] = real
                    mftData[n + 1] = imag
            else:
                mftData = self.DFTunnormed(series[0:self.windowSize])

            normalisingFactor = (1 / stds[i] if stds[i] > 0 else 1) * self.inverseSqrtWindowSize;
            transformed[i] = mftData * normalisingFactor;

        return transformed

    def calcIncrementalMeanStd(self, series, end):
        means = np.zeros(end)
        stds = np.zeros(end)

        sum = 0
        squareSum = 0

        for ww in range(self.windowSize):
            sum += series[ww]
            squareSum += series[ww] * series[ww]

        rWindowLength = 1 / self.windowSize
        means[0] = sum * rWindowLength
        buf = squareSum * rWindowLength - means[0] * means[0]
        stds[0] = math.sqrt(buf) if buf > 0 else 0

        for w in range(1, end):
            sum += series[w + self.windowSize - 1] - series[w - 1]
            means[w] = sum * rWindowLength
            squareSum += series[w + self.windowSize - 1] * series[w + self.windowSize - 1] - series[w - 1] * series[w - 1]
            buf = squareSum * rWindowLength - means[w] * means[w]
            stds[w] = math.sqrt(buf) if buf > 0 else 0

        return stds

    def createWord(self, dft):
        word = BitWord()

        for i in range(self.wordLength):
            for bp in range(self.alphabetSize):
                if dft[i] <= self.breakpoints[i][bp]:
                    word.push(bp)
                    break

        return word

    def shorten_bags(self, wordLen):
        newBags = []

        for i in range(self.num_insts):
            bag = {}
            lastWord = -1

            for n, word in enumerate(self.words[i]):
                newWord = BitWord(word = word.word, length = word.length)
                newWord.shorten(16 - wordLen)
                lastWord = self.addToBag(bag, newWord, lastWord)

            newBags.append(bag)

        return newBags;

    def addToBag(self, bag, word, lastWord):
        if word.word == lastWord:
            return lastWord

        if word.word in bag:
            bag[word.word] += 1
        else:
            bag[word.word] = 1

        return word.word

class BitWord:

    def __init__(self, word = np.int_(0), length = 0):
        self.word = word
        self.length = length

    def push(self, letter):
        self.word = (self.word << 2) | letter
        self.length += 1

    def shorten(self, amount):
        self.word = self.rightShift(self.word,amount*2)
        self.length -= amount

    def wordList(self):
        wordList = []
        shift = 32-(self.length*2)

        for i in range(self.length-1, -1, -1):
            wordList.append(self.rightShift(self.word << shift, 32-2))
            shift += 2

        return wordList

    def rightShift(self, left, right):
        return (left % 0x100000000) >> right

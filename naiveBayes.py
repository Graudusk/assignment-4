import math
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import norm


class naiveBayes:
    means = dict()
    stds = dict()

    def __init__(self):
        self.means.clear()
        self.stds.clear()

    # Runs n-fold cross-validation and returns a list of predictions
    def crossval_predict(self, x, y, folds):
        preds = []
        accScores = []
        kf = KFold(n_splits=folds, shuffle=True)
        # Run k-fold cross-validation iterations
        for train_indexes, test_indexes in kf.split(x):
            # Split the data into k-folds
            train_subset = []
            train_labels = []
            test_subset = []
            test_labels = []
            for i in train_indexes:
                train_subset.append(x[i])
                train_labels.append(y[i])
            for j in test_indexes:
                test_subset.append(x[j])
                test_labels.append(y[j])
            # Train the model
            self.fit(train_subset, train_labels)
            # Test the model
            preds = self.predict(test_subset)
            # Calculate the accuracy
            acc = self.accuracy_store(preds, test_labels)
            print("Accuracy: " + str(round(acc * 100, 2)) + "%\n")
            accScores.append(acc)
        return accScores

    # Trains the model on input examples X and labels Y
    def fit(self, x, y):
        # Calculate mean value and standard deviation of each attribute for each category

        labels = []
        length = len(x)
        rowLength = len(x[0])

        for i in range(len(y)):
            if y[i] not in labels:
                labels.append(y[i])

        for i in range(len(labels)):
            self.means[labels[i]] = [0.0, 0.0, 0.0, 0.0]
            self.stds[labels[i]] = [0.0, 0.0, 0.0, 0.0]

        for i in range(length):
            label = y[i]
            for o in range(rowLength):
                self.means[label][o] += float(x[i][o])

        subsetLength = length / len(self.means.keys())
        for c in self.means:
            for o in range(rowLength):
                self.means[c][o] = self.means[c][o] / subsetLength

        for i in range(length):
            label = y[i]
            for o in range(rowLength):
                self.stds[label][o] += (float(x[i][o]) - self.means[label][o]) ** 2

        for c in self.stds:
            for o in range(rowLength):
                self.stds[c][o] = math.sqrt(self.stds[c][o] / subsetLength)

    # Classifies examples X and returns a list of predicitions
    def predict(self, x):
        preds = []
        # Calculate the probabibities of each attribute belonging to each category
        for newExample in x:
            # Get current example value, mean value for each category and standard deviation for each category
            p = dict()
            for i in range(len(self.means)):
                p[i] = []
            for i in range(len(newExample)):
                value = float(newExample[i])
                meanValues = []
                stdValues = []
                for row in self.means:
                    meanValues.append(float(self.means[row][i]))
                for row in self.stds:
                    stdValues.append(float(self.stds[row][i]))
                # calculate the probabilities of the input attributes belonging to each category using the Gaussian Probability Density Function
                for j in range(len(meanValues)):
                    # pdf(xi,meani, stdi) = (1 / (sqrt(2 * PI) * stdi)) * e^(-((xi-meani)^2)/(2 * stdi^2)))
                    # combine the log of the probabilities together
                    p[j].append(norm.logpdf(
                        value, meanValues[j], stdValues[j]))
            s = []
            for c in p:
                # Transform the equivalent product ln(xy) back into the original form:eln(xy)
                p[c] = math.exp(p[c][0] + p[c][1] + p[c][2] + p[c][3])
                s.append(p[c])
            total = 0.0
            for sum in s:
                total += sum
            # Normalize each probability and classify the example as the category with the highest probability
            highestP = 0.0
            catHighestP = 0
            for c in p:
                if (p[c] / total) > highestP:
                    highestP = (p[c] / total)
                    catHighestP = c
            preds.append(catHighestP)
        return preds

    # Calculates accuracy store from a list of predictions
    def accuracy_store(self, preds, y):
        print(self.confusion_matrix(preds, y))
        return accuracy_score(y, preds)

    # Generate confusion matrix
    def confusion_matrix(self, preds, y):
        conf_matrix = confusion_matrix(y, preds)
        return conf_matrix

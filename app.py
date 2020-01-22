from naiveBayes import naiveBayes
from data import data

folds = 8

naiveBayesIris = naiveBayes()
print("Iris Dataset\n===================================\n")
# Get Iris data
irisData = data.read('./db/Iris/iris.csv')
# Train and test the model with Iris data
irisAccuracyScores = naiveBayesIris.crossval_predict(irisData["data"], irisData["labels"], folds)
# Calculate accuracy score
irisTotalAccuracy = 0.0
for i in irisAccuracyScores:
    irisTotalAccuracy += i
irisTotalAccuracy = irisTotalAccuracy / len(irisAccuracyScores)
print("\n\nTotal accuracy: " +
      str(round(irisTotalAccuracy * 100, 2)) + "%\n\n")


naiveBayesbanknote = naiveBayes()
print("Banknote Dataset\n===================================\n")
# Get Iris data
banknoteData = data.read('./db/banknote_authentication/banknote_authentication.csv')
# Train and test the model with banknote data
banknoteAccuracyScores = naiveBayesbanknote.crossval_predict(banknoteData["data"], banknoteData["labels"], folds)
# Calculate accuracy score
banknoteTotalAccuracy = 0.0
for i in banknoteAccuracyScores:
    banknoteTotalAccuracy += i
banknoteTotalAccuracy = banknoteTotalAccuracy / len(banknoteAccuracyScores)
print("\n\nTotal accuracy: " +
      str(round(banknoteTotalAccuracy * 100, 2)) + "%\n\n")

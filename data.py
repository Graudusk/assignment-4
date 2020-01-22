import csv
from csv import reader

class data:
    def read(path):
        data = dict()
        with open(path, 'r') as csvfile:
            first_line = csvfile.readline()
            keys = first_line.rstrip().split(',')
            line = 0
            reader = csv.reader(csvfile)
            data["data"] = []
            data["labels"] = []
            data["label_names"] = []
            data["feature_names"] = []
            for x in keys:
                data["feature_names"].append(x)
            for row in reader:
                data["data"].append([row[0], row[1], row[2], row[3]])
                if not row[4] in data["label_names"]:
                    data["label_names"].append(row[4])
                data["labels"].append(
                    data["label_names"].index(row[4]))
        return data

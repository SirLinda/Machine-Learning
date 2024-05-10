import pandas as pd
from sklearn.linear_model import Perceptron

def perceptron(x,weights,threshold):
    weight_sum = 0
    # Calculate sum
    for x, w in zip(x, weights):
        weight_sum += x * w
        # print(weight_sum)

    if weight_sum > threshold:
        return 1
    else:
        return 0

# initial weights, threshold, learning rate
weights = [2.5, -3, 1.5]
threshold = 2
eta = 0.05

# read csv file
data = pd.read_csv('Percept1.csv', delimiter='\t')
print(data)


# epoch
for row in data:

    row = row.split(';')
    data_point = [float(row[0]), float(row[1]), float(row[2])]

    x = (data_point[0], data_point[1])

    t = float(data_point[2])

    y = perceptron(x, weights, threshold)
    print(y)

    for i in range(len(weights)):
        weights[i] += eta * (t - y) * data_point[i]
    threshold -= eta * (t - y)
    print(weights)


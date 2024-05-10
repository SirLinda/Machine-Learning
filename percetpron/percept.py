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

# Initialize
weights = [2.5, -3, 1.5]
threshold = 2
x = []

# take in x value
for i in range(len(weights)):
    x_value = float(input())
    x.append(x_value)

# output
output = perceptron(x,weights,threshold)
print(str(output))

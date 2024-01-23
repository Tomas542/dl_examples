input_value = 0.8
goal_value = 1
weight = 0.2
learning_rate = 0.1

for epoch in range(100):
    pred_value = input_value * weight
    error = (pred_value - goal_value) ** 2
    derivative = (pred_value - goal_value) * input_value
    weight = weight - derivative * learning_rate
    print(f"Error is {error}, prediction is {pred_value}")
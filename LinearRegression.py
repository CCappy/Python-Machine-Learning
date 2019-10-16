from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random 

style.use('fivethirtyeight')

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

def create_dataset(hm, variance, step=2, correlation=False): #hm = how many data points, variance = how variable do we want this dataset to be, step = slope
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
    
def best_fit_slope_and_intercept(xs, ys):
    m = ((mean(xs) * mean(ys) - mean(xs*ys)) /
        (mean(xs)**2 - mean(xs**2)))
    b = mean(ys) - m*mean(xs) 
    return m, b

def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig] # makes a single value, which is the mean of y for every y we have
    squared_error_reqr = squared_error(ys_orig, ys_line) #Top of our equation
    squared_error_y_mean = squared_error(ys_orig, y_mean_line) #Bottom of our equation
    return 1 - (squared_error_reqr / squared_error_y_mean) # coef of determimation equation

xs, ys = create_dataset(40, 5, 2, correlation='neg')
m, b = best_fit_slope_and_intercept(xs, ys)

regression_line = [(m * x) + b for x in xs] #y = mx + b, append each y to regression line

predict_x = 8 # These two lines predict what y will be when x is 8
predict_y = (m*predict_x) + b

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y) #Point for y = m*8 + b
plt.plot(xs, regression_line)# Plots our regression line
plt.show()

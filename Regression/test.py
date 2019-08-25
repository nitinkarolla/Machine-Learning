
import numpy as np

m = 200
w = 1
b = 5
var = 0.1

def check_parameters(w,b,m,var) :
    w_org = []
    b_org = []
    w_dash = []
    b_dash = []
    for i in range(1000):
        x = list(np.random.uniform(low=100,high =102, size=m))
        x_dash = list(map(lambda j : j - 101, x))
        y_temp = [((w*j) + b) for j in x]
        e = list(np.random.normal(0,scale= np.sqrt(var), size = m))
        y = [a + b for a, b in zip(y_temp, e)]
        w_org.append(np.cov(x,y)[0][1]/ np.var(x))
        b_org.append(np.mean(y) - w * np.mean(x))
        w_dash.append(np.cov(x_dash,y)[0][1]/ np.var(x_dash))
        b_dash.append(np.mean(y) - w * np.mean(x_dash))
    print("The expected value of w is : ", np.mean(w_org))
    print("The variance of w is : ", np.var(w_org))
    print("The expected value of b is : ", np.mean(b_org))
    print("The variance of b is : ", np.var(b_org))
    print("The expected value of w_dash is : ", np.mean(w_dash))
    print("The variance of w_dash is : ", np.var(w_dash))
    print("The expected value of b_dash is : ", np.mean(b_dash))
    print("The variance of b_dash is : ", np.var(b_dash))

check_parameters(w,b,m,var)
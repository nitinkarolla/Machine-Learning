import numpy as np
import time
import math

class DualSVM():
    
    def __init__(self, 
                 problem = "dual", 
                 fit_intercept = True, 
                 tol = 0.001, 
                 tau = 1, 
                 mu = 15, 
                 t_0 = 1, 
                 use_kernel = True):
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.tau = tau
        self.mu = mu
        self.t_0 = t_0
        self.use_kernel = use_kernel
        self.problem = problem

    def log_barrier(self,x):
        x = -np.sum(np.log(x))
        return x
        
    def _phi(self, x, t):
        x = t * (1/2*np.dot(x, self.Q.dot(x)) + self.p.dot(x)) + self.log_barrier(self.b - self.A.dot(x))
        return x
    def _grad(self, x, t):
        x = t*(self.Q.dot(x) + self.p) + np.sum(np.divide(self.A.T, self.b - self.A.dot(x)), axis = 1)
        return x
    def _hess(self, x, t):
        A_ = np.divide(self.A.T, self.b - self.A.dot(x))
        x = t*self.Q + A_.dot(A_.T)
        return x
    
    def _poly_kernel(self, X, n, d = 3):
        return (np.identity(n = n) + X.T.dot(X))**d
        
    def _convert_to_primal(self, X, y):
        # Features' dimension (d+1)
        d_ = X.shape[0]
        d = d_ - 1
        # Number of observations
        n = X.shape[1]
        # Multiply each observation (xi) by its label (yi)
        X_ = X*y
        # Shape (d+1+n, d+1+n)
        Q = np.zeros((d_ + n, d_ + n))
        Q[:d, :d] = np.identity(d)
        # Shape (d+1+n,)
        p = np.zeros(d_ + n)
        p[d_:] = 1/(self.tau*n)
        # Shape (2*n, d+1+n)
        A = np.zeros((2*n, d_+n))
        A[:n, :d_] = -X_.T
        A[:n, d_:] = np.diag([-1]*n)
        A[n:, d_:] = np.diag([-1]*n)
        # Shape (2*n, )
        b = np.zeros(2*n)
        b[:n] = -1
        
        return Q, p, A, b
    
    def _convert_to_dual(self, X, y):
        
        # Number of observations
        n = X.shape[1]
        
        # Use a kernel or no kernel
        if self.use_kernel:
            Q = y*y
            K = self._poly_kernel(X = X, n = n)
            Q = Q*K
        else:
            X_ = X*y
            Q = X_.T.dot(X_)
        # Shape (n, )
        p = -np.ones(n)
        # Shape (2*n, n)
        A = np.zeros((2*n, n))
        A[:n, :] = np.identity(n)
        A[n:, :] = -np.identity(n)
        # Shape (2*n, )
        b = np.zeros(2*n)
        b[:n] = 1/(self.tau*n)
        return Q, p, A, b
    
    def _damped_newton_step(self, x, obj_function, gradient, hessian):
        g = gradient(x)
        h = hessian(x)
        h_inv = np.linalg.inv(h)
        lambda_ = (g.T.dot(h_inv.dot(g)))**(1/2)
        xnew = x - (1/(1+lambda_))*h_inv.dot(g)
        gap = 1/2*lambda_**2
        return xnew, gap
    
    def _damped_newton(self, x, obj_function, gradient, hessian):
        
        # First step
        x, gap = self._damped_newton_step(x, obj_function, gradient, hessian)
        xhist = [x]
        # For theoritical reasons, tol should be smaller than (3-sqrt(5))/2
        if self.tol < (3 - np.sqrt(5))/2:
            while gap > self.tol:
                x, gap = self._damped_newton_step(x, obj_function, gradient, hessian)
                xhist.append(x)
            xstar = x
        else:
            raise ValueError("Enter a value for tol < (3-sqrt(5))/2")
        return xstar, xhist
    
    def _solve_barrier_problem(self, alphas_0):
        
        outer_iterations = []
        m = self.b.shape[0]
        if np.sum(self.A.dot(alphas_0) < self.b) == m:
            t = self.t_0
            x = alphas_0
            xhist = [alphas_0]
            while m/t >= self.tol:
                
                obj_function = lambda x: self._phi(x, t)
                gradient = lambda x: self._grad(x, t)
                hessian = lambda x: self._hess(x, t)
                x, xhist_newton = self._damped_newton(x, obj_function, gradient, hessian)
                xhist += xhist_newton
                outer_iterations += [len(xhist_newton)]
                t *= self.mu      
                print("Outer iteration number {} completed".format(len(outer_iterations)))
            self.x_sol = x
        else:
            raise ValueError("x_0 is not scritly feasible, cannot proceed")
        return xhist, outer_iterations 

    def fit(self, X, y):
        
        n, d = X.shape
        X = np.vstack((X.T, np.ones(n)))
        
        # Convert problem
        if self.problem == "dual":
            alphas_0 = (1/(100*self.tau*n))*np.ones(n)
            self.Q, self.p, self.A, self.b = self._convert_to_dual(X, y)
        else:
            # Strictly feasible point for the primal
            alphas_0 = np.zeros(d + 1 + n)
            alphas_0[d + 1:] = 1.1
            self.Q, self.p, self.A, self.b = self._convert_to_primal(X, y)
            
        # Solve the barrier optimisation problem
        x_hist, outer_iterations = self._solve_barrier_problem(alphas_0)
        
        # Calculate the weights
        if self.problem == "dual":
            self.w = self.x_sol.dot((X*y).T)
        else:
            self.w = self.x_sol[:d + 1]
    
    def predict(self, X_test, y_test):
        # Number of training examples
        self.n_test = X_test.shape[0]
        # Add offset to the data points and make X of shape (d+1, n)
        X_test = np.vstack((X_test.T, np.ones(self.n_test)))
        # Predict
        y_pred = np.sign(self.w.T.dot(X_test))
        # Compute the mean accuracy on the predictions
        accuracy = self.compute_mean_accuracy(y_pred, y_test)
        return y_pred, accuracy

    def compute_mean_accuracy(self, y_pred, y_test):
        accuracy = np.sum(y_pred == y_test)
        accuracy /= np.shape(y_test)[0]
        return accuracy


from sklearn.datasets import load_iris
iris = load_iris()

svm = DualSVM()
svm.fit(iris.data, iris.target)
print(svm.predict(iris.data, iris.target)[1])
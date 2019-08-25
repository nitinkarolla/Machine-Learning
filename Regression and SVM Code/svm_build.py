import numpy as np
import time
import math


class SVM():
    def __init__(self, tau = 1, t_0 = 1, tol = 0.0001, mu = 15, 
                 solution = "Dual", kernel = None,
                 poly_d = 2):
        self.tau = tau
        self.t_0 = t_0
        self.tol = tol
        self.mu = mu
        self.kernel = kernel
        self.poly_d = poly_d
        self.solution = solution
    
    def poly_kernel(self, X, n, d = 2):
        return (np.identity(n = n) + X.T.dot(X))**d
    
    def log_barrier(self,x):
        x = -np.sum(np.log(x))
        return x

    def phi(self, x, t, Q, p, A, b):
        """ Compute the value of the function of x:
        phi = t(1/2*xTQx + pTx) + B(b - Ax) 
        where Q, p, A, b are the matrices of the quadratic problem:
        minimize(x) 1/2*xTQx + pTx 
        st. Ax <= b """
        x = t * (1/2*np.dot(x, Q.dot(x)) + p.dot(x)) + self.log_barrier(b-A.dot(x))
        return x

    def grad(self, x, t, Q, p, A, b):
        """ Compute the gradient of the function of x:
        phi = t(1/2*xTQx + pTx) + B(b - Ax) 
        where Q, p, A, b are the matrices of the quadratic problem:
        minimize(x) 1/2*xTQx + pTx 
        st. Ax <= b"""
        x = t*(Q.dot(x) + p) + np.sum(np.divide(A.T, b-A.dot(x)), axis = 1)
        return x

    def hess(self, x, t, Q, p, A, b):
        """ Compute the hessian of the function of x:
        phi = t(1/2*xTQx + pTx) + B(b - Ax) 
        where Q, p, A, b are the matrices of the quadratic problem:
        minimize(x) 1/2*xTQx + pTx 
        st. Ax <= b"""
        # Divide each column i of A is divided by bi- (Ax)i
        A_ = np.divide(A.T, b-A.dot(x))
        x = t*Q + A_.dot(A_.T)
        return x
    
    def transform_svm_dual(self,tau, X, y):
        # Number of observations
        n = X.shape[1]
        # Multiply each observation (xi) by its label (yi)
        if self.kernel == 'Polynomial':
            Q = y*y
            K = self.poly_kernel(X = X, n = n, d = self.poly_d)
            Q = Q*K
        else:
            X_ = X*y
            Q = X_.T.dot(X_)
        p = -np.ones(n)
        # Shape (2*n, n)
        A = np.zeros((2*n, n))
        A[:n, :] = np.identity(n)
        A[n:, :] = -np.identity(n)
        # Shape (2*n, )
        b = np.zeros(2*n)
        b[:n] = 1/(tau*n)
        return Q, p, A, b

    def transform_svm_primal(self,tau, X, y):
        d_ = X.shape[0]
        d = d_ - 1
        n = X.shape[1]
        X_ = X*y
        Q = np.zeros((d_+n, d_+n))
        Q[:d, :d] = np.identity(d)
        p = np.zeros(d_+n)
        p[d_:] = 1/(tau*n)
        A = np.zeros((2*n, d_+n))
        A[:n, :d_] = -X_.T
        A[:n, d_:] = np.diag([-1]*n)
        A[n:, d_:] = np.diag([-1]*n)
        b = np.zeros(2*n)
        b[:n] = -1
        return Q, p, A, b
    
    def NewtonStep(self, x, f, g, h):
        """ Compute the Newton step at x for the function f.
        """
        g = g(x)
        h = h(x)
        h_inv = np.linalg.inv(h)
        lambda_ = (g.T.dot(h_inv.dot(g)))**(1/2)

        newton_step = -h_inv.dot(g)
        gap = 1/2*lambda_**2

        return newton_step, gap

    def backTrackingLineSearch(self, x, step, f, g, A, b, alpha = 0.3, beta = 0.5):
        """ Compute the step size minimizing(t) f(x + t*step) with 
        backtracking line-search.
        """
        step_size = 1
        m = b.shape[0]
        xnew = x + step_size*step
        while np.sum(A.dot(xnew) < b) < m:
            step_size *= beta
            xnew = x + step_size*step
        y = f(xnew)
        z = f(x) + alpha*step_size*g(x).T.dot(step)
        while y > z:
            step_size *= beta
            xnew = x + step_size*step
            y = f(xnew)
            z = f(x) + alpha*step_size*g(x).T.dot(step)
        return step_size

    def newtonLS(self, x0, f, g, h, tol, A, b, alpha = 0.3, beta = 0.5):
        newton_step, gap = self.NewtonStep(x0, f, g, h)
        step_size = self.backTrackingLineSearch(x0, newton_step, f, g, A, b, alpha, beta)
        x = x0 + step_size*newton_step
        xhist = [x]
        while gap > tol:
            newton_step, gap = self.NewtonStep(x, f, g, h)
            step_size = self.backTrackingLineSearch(x, newton_step, f, g, A, b, alpha, beta)
            x += step_size*newton_step
            xhist.append(x)
        xstar = x
        return xstar, xhist

    def barr_method(self, Q, p, A, b, x_0, t_0, mu, tol):
        outer_iterations = []
        m = b.shape[0]
        if np.sum(A.dot(x_0) < b) == m:
            t = t_0
            x = x_0
            xhist = [x_0]
            while m/t >= tol:
                f = lambda x: self.phi(x, t, Q, p, A, b)
                g = lambda x: self.grad (x, t, Q, p, A, b)
                h = lambda x: self.hess (x, t, Q, p, A, b)
                x, xhist_Newton = self.newtonLS(x, f, g, h, tol, A, b)
                xhist += xhist_Newton
                outer_iterations += [len(xhist_Newton)]
                t *= mu      
                print("Outer iteration number {} completed".format(len(outer_iterations)))
            x_sol = x
        else:
            raise ValueError("x_0 is not scritly feasible, cannot proceed")
        return x_sol, xhist, outer_iterations 

    def train(self, X, y):
        self.n = X.shape[0]
        self.d = X.shape[1]
        X = np.vstack((X.T, np.ones(self.n)))
        if self.solution == 'Dual':
            self.x_0 = (1/(100*self.tau*self.n))*np.ones(self.n)
            self.Q, self.p, self.A, self.b = self.transform_svm_dual(self.tau, X, y)
            self.x_sol, self.xhist, self.outer_iterations = self.barr_method(self.Q, 
                self.p, self.A, self.b, self.x_0, self.t_0, self.mu, self.tol)
            self.w = self.x_sol.dot((X*y).T)
        elif self.solution == 'Primal':
            self.x_0 = np.zeros(self.d+1+self.n)
            self.x_0[self.d + 1:] = 1.1
            self.Q, self.p, self.A, self.b = self.transform_svm_primal(self.tau, X, y)
            self.x_sol, self.xhist, self.outer_iterations = self.barr_method(self.Q, 
                self.p, self.A, self.b, self.x_0, self.t_0, self.mu, self.tol)
            self.w = self.x_sol[:self.d + 1]

    def predict(self, X_test, y_test):
        self.n_test = X_test.shape[0]
        X_test = np.vstack((X_test.T, np.ones(self.n_test)))
        y_pred = np.sign(self.w.T.dot(X_test))
        accuracy = self.compute_mean_accuracy(y_pred, y_test)
        return y_pred, accuracy

    def compute_mean_accuracy(self, y_pred, y_test):
        accuracy = np.sum(y_pred == y_test)
        accuracy /= np.shape(y_test)[0]
        return accuracy

from sklearn.datasets import load_iris
iris = load_iris()

svm = SVM()
svm.train(iris.data, iris.target)
print(svm.predict(iris.data, iris.target)[1])

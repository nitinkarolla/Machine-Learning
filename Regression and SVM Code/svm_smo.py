import numpy as np

def linear_kernel(x, y, b=1):
    return x @ y.T + b

def gaussian_kernel(x, y, sigma=1):
    if np.ndim(x) == 1 and np.ndim(y) == 1:
        result = np.exp(- (np.linalg.norm(x - y, 2)) ** 2 / (2 * sigma ** 2))
    elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
        result = np.exp(- (np.linalg.norm(x - y, 2, axis=1) ** 2) / (2 * sigma ** 2))
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
        result = np.exp(- (np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2) ** 2) / (2 * sigma ** 2))
    return result

class SupportVectorMachines():
    def __init__(self, X= None, Y= None, max_iterations = 50, epsilon = 0.001, C= 1.0, kernel = None,
                 min_optimisation_stop = 0.000001):
        self.X = X
        self.Y = Y
        self.alpha = np.mat(np.zeros((X.shape[0],1)))
        self.b = np.mat([[0]])
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.C = C
        self.min_optimisation_stop = min_optimisation_stop
        
        if kernel == None:
            self.kernel = linear_kernel
        elif kernel == 'Gaussian':
            self.kernel = gaussian_kernel

    
    def prepare_data(self):
        return
    
    def perform_smo(self):
        alpha_pairs_optimised = 0
        for i in range(self.X.shape[0]):
            Ei = np.multiply(self.Y, self.alpha).T * self.kernel(self.X ,self.X[i]) + self.b - self.Y[i]
            if (self.check_kkt(self.alpha[i], Ei)):
                j = self.second_alpha_to_opt(i,self.X.shape[0])
                Ej = np.multiply(self.Y, self.alpha).T * self.kernel(self.X , self.X[j]) + self.b - self.Y[j]
                alphaoldi = self.alpha[i].copy()
                alphaoldj = self.alpha[j].copy()
                bounds = self.alpha_bounding(self.alpha[i], self.alpha[j], self.Y[i], self.Y[j])
                k11 = self.kernel(self.X[i], self.X[i])
                k12 = self.kernel(self.X[i], self.X[j])
                k22 = self.kernel(self.X[j], self.X[j])
                n_eta = 2 * k12 - k11 - k22
                if bounds[0] != bounds[1] and n_eta <0 :
                    if self.optimize_alpha_pair( i, j, Ei, Ej, n_eta, bounds, alphaoldi, alphaoldj):
                        alpha_pairs_optimised = alpha_pairs_optimised + 1
        return alpha_pairs_optimised
    
    def check_kkt(self, alpha, E):
        return (alpha > 0 and np.abs(E) < self.epsilon) or (alpha < self.C or np.abs(E) > self.epsilon)
        
    def second_alpha_to_opt(self, indexof1alpha, nrows):
        indexof2alpha = indexof1alpha
        while (indexof1alpha == indexof2alpha):
            indexof2alpha = int(np.random.uniform(0,nrows))
            return indexof2alpha
    
    def alpha_bounding(self, alpha_i, alpha_j, y_i, y_j):
        bounds = [2]
        if (y_i == y_j):
            bounds.insert(0, max(0,alpha_j + alpha_i -self.C))
            bounds.insert(1, min(self.C,alpha_j + alpha_i))
        else:
            bounds.insert(0, max(0,alpha_j - alpha_i ))
            bounds.insert(1, min(self.C,alpha_j - alpha_i + self.C))
        return bounds
    
    def optimize_alphai_alphaj(self, i , j, alpha_j_old):
        self.alpha[i] = self.alpha[j] * self.Y[i] *(alpha_j_old - self.alpha[j])

    def clip_alpha_j(self, j, bounds):
        if self.alpha[j] < bounds[0] :
            self.alpha[j] = bounds[0]
        if self.alpha[j] > bounds[1]:
            self.alpha[j] = bounds[1]

    def optimize_alpha_pair(self, i, j, Ei, Ej, n_eta, bounds, alphaoldi, alphaoldj):
        flag = False
        self.alpha[j] = self.alpha[j] - self.Y[j] * (Ei - Ej) / n_eta
        self.clip_alpha_j(j, bounds)
        if (abs(self.alpha[j] - alphaoldj) >= self.min_optimisation_stop):
            self.optimize_alphai_alphaj(i,j,alphaoldj)
            self.optimize_b(i, j, Ei, Ej, alphaoldi, alphaoldj)
            flag = True
        return flag
    
    def optimize_b(self,i, j, Ei, Ej, alphaoldi, alphaoldj):
        b1 = self.b - Ei - self.Y[i] * (self.alpha[i] - alphaoldi) * self.kernel(self.X[i],self.X[i]) \
                         - self.Y[j] * (self.alpha[j] - alphaoldj) * self.kernel(self.X[i],self.X[j])
        b2 = self.b - Ej - self.Y[i] * (self.alpha[i] - alphaoldi) * self.kernel(self.X[i],self.X[j]) \
                         - self.Y[j] * (self.alpha[j] - alphaoldj) * self.kernel(self.X[j] ,self.X[j])
        if (0 < self.alpha[i] and (self.C > self.alpha[i])):
            self.b = b1
        elif (0 < self.alpha[j] and (self.C > self.alpha[j])):
            self.b = b2
        else:
            self.b = (b1 + b2)/2.0
    
    def calculate_w(self, alpha, X, Y):
        w = np.zeros((X.shape[1],1))
        for i in range(X.shape[0]):
            w = w + np.multiply(Y[i] * alpha[i], X[i].T)
        return w
        
    def fit(self):
        i = 0
        while (i < self.max_iterations):
            if (self.perform_smo() == 0):
                i = i+1
            else:
                i = 0
        self.W = self.calculate_w(self.alpha, self.X, self.Y)

    def predict_row(self, x):
        classification = 0
        if (np.sign((x @ self.W + self.b).item(0,0)) == 1):
            classification = 1
        return classification

            


data = [
    [[9.12, 3.12], [+1]],
    [[9.12, 5.12], [+1]],
    [[5.12, 5.12], [-1]],
    [[8.12, 6.65], [+1]],
    [[4.65, 4.12], [-1]]
    ]

X = []
Y = []

for i in range(len(data)):
    X.append(data[i][0])
    Y.append(data[i][1])

svm = SupportVectorMachines(X= np.mat(X), Y= np.mat(Y))
svm.fit()

print(svm.W)
print(svm.b)
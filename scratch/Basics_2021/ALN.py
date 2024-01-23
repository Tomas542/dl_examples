import numpy as np

class AdaptineLinearNeuron():
    def __init__(self, rate:float = 0.01, niter:int = 10) -> None:
        self.rate = rate
        self.niter = niter

    def fit(self, X:np.ndarray, y:float):
        self.weight = np.zeros(1 + X.shape[1])
        self.cost = 0
        
        for i in range(self.niter):
            output = self.net_input(X)
            errors = y - output
            self.weight[1:] += self.rate * X.T.dot(errors)
            self.weight[0] += self.rate * errors.sum()
            cost = (errors ** 2).sum() / 2
            self.cost.append(cost)

        return self
    
    def net_input(self, X):
        return np.dot(X, self.weight[1:]) + self.weight[0]
    
    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

class Perceptron(object):
    def init (self, eta: float = 0.01, n_iter:int = 10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.w = np.zeros(1 + X.shape[1])
        self.errors = []
        
        for _ in range(self.n_iter):
            errors = 0
        
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            
            self.errors .append(errors)
    
        return self

    def net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.net_input(X) >= 0.0, 1, -1)

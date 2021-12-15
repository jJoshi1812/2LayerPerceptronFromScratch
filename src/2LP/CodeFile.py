import numpy as np
import pandas as pd
import csv
import Loading_Data
import warnings
warnings.filterwarnings('ignore')

#Code for Perceprton

class TLP:
    def __init__(self, alpha):
        self.__alpha = alpha
        self.__layers = []
    
    def add(self, layer):
        self.__layers.append(layer)

    def fit(self, x, y):
        self.__inpt = x
        self.__y = y

    def getAlpha(self):
        return self.__alpha

    def getLayers(self):
        return self.__layers

    def compile(self, epochs=10):
        dims = self.__inpt.shape[0]
        for i in range(len(self.__layers)):
            self.__layers[i].initializeLayer(dims)
            dims = self.__layers[i].getNodes()
        print('Initialization of layers completed.')
        self.gradientDescent(epochs)

    def forwardPropagation(self, inpt):
        layers = self.__layers
        for i in range(len(layers)):
            layer = layers[i]
            inpt = layer.forwardStep(inpt)
        return inpt

    def backwardPropagation(self, derivativeA):
        layers = self.__layers
        for i in range(len(layers) - 1, -1, -1):
            layer = layers[i]
            derivativeA = layer.backStep(derivativeA, self.__alpha)

    def gradientDescent(self, epochs):
        for i in range(epochs):
            for j in range(self.__inpt.shape[1]):
                inpt = self.__inpt[:, j].reshape(-1, 1)
                a = self.forwardPropagation(inpt)
                y_cap = self.__y[:, j].reshape(-1, 1)
                lossDerivative = 2 * np.subtract(a, y_cap)
                self.backwardPropagation(lossDerivative)
            print('Epoch ', i + 1, ' COMPLETED')

    def predict(self, X_test):
        pred = []
        for i in range(X_test.shape[1]):
            inpt = X_test[:, i].reshape(-1, 1)
            a = self.forwardPropagation(inpt)
            pred.append(np.argmax(a))
        return pred

# Base class for Layers
class Layers:
    def __init__(self, n):
        self.__n = n
        self.__weights = None
        self.__bias = None

    def initializeLayer(self, inputdims):
        np.random.seed(35)
        self.__weights = np.random.rand(self.__n, inputdims) * 10 ** (-9)
        self.__bias = np.zeros((self.__n, 1))

    def getNodes(self):
        return self.__n

    def getWeights(self):
        return self.__weights

    def setWeights(self,weights):
        self.__weights = weights
    
    def getBias(self):
        return self.__bias

    def setBias(self,bias):
        self.__bias = bias

    def backStep(self, derivativeA, learningRate):
        z = self.__z.copy()
        a = self.__inpt.copy()
        weights = self.__weights.copy()
        derivativeZ = np.multiply(self.derivativeActivation(z), derivativeA).copy()
        derivativeWeights = (derivativeZ @ a.T).copy()
        derivativeBias = derivativeZ.copy()
        derivativeAprev = (weights.T @ derivativeZ).copy()

        self.__weights = np.subtract(self.__weights, learningRate * derivativeWeights).copy()
        self.__bias = np.subtract(self.__bias, learningRate * derivativeBias).copy()
        return derivativeAprev

    def forwardStep(self, inpt):
        self.__inpt = inpt.copy()
        self.__z = np.add((self.__weights @ self.__inpt), self.__bias).copy()
        a = self.activation(self.__z).copy()
        return a

# subclass of a Layer class (Inheritance)
class LayerRelu(Layers):
    def __init__(self, n):
        super().__init__(n)

    def activation(self, Z):
        return (np.maximum(0, Z))

    def derivativeActivation(self, Z):
        R = self.activation(Z)
        # print(Z)
        return np.where(R > 0, 1, 0)

# Another subclass
class LayerSigmoid(Layers):
    def __init__(self, n):
        super().__init__(n)

    def activation(self, z):
        return 1 / (1 + np.exp(-z))

    def derivativeActivation(self, z):
        s = self.activation(z)
        return s * (1 - s)


def creating_csv_file(header, data, name):
    with open(name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write multiple rows
        writer.writerows(data)


if __name__=="__main__":
    
    # loading the database
    db=Loading_Data.DatabaseLoader()
    db.create_table()
    data=Loading_Data.DataFrameLoader()
    dataset_x_tr, dataset_x_ts,dataset_y_tr, dataset_y_ts=data.load_dataframes()
    dataset_y_tr.drop(['index'],inplace=True,axis=1)
    dataset_x_tr.drop(['index'],inplace=True,axis=1)
    dataset_y_ts.drop(['index'],inplace=True,axis=1)
    dataset_x_ts.drop(['index'],inplace=True,axis=1)
    

    # Features
    X_train = dataset_x_tr

    # Labels
    y_train = dataset_y_tr
    print('\nDataSet has been split into train and Validation set! 10% of data will be used as Validation Set')


    
    y = pd.get_dummies(dataset_y_tr.squeeze())
    y = y.values.T
    X = dataset_x_tr.values.reshape(60000, 784).T

    # feeding the data into 2 Layer Perceptron
    model = TLP(0.001)
    model.add(LayerRelu(5))
    model.add(LayerSigmoid(10))
    model.fit(X, y)
    model.compile(epochs=3)

    # Checking the accuracy
    X_test=dataset_x_ts.values.T
    y_pred=model.predict(X_test)
    from sklearn.metrics import accuracy_score
    acc=accuracy_score(dataset_y_ts.values, y_pred)
    print(f'The accuracy on test set is {round(acc*100,2)}%')
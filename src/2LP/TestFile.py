import sys
import gc
import unittest
import Loading_Data
import numpy as np
from sklearn import model_selection
import pandas as pd
import CodeFile

class Test_MLP(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print("Loading the dataset and getting X (i.e features) and y (i.e target)")
        data = Loading_Data.DataFrameLoader()
        print('Loading DataSet Done!')

        dataset_x_tr, dataset_x_ts,dataset_y_tr, dataset_y_ts=data.load_dataframes()
        dataset_y_tr.drop(['index'],inplace=True,axis=1)
        dataset_x_tr.drop(['index'],inplace=True,axis=1)
        dataset_y_ts.drop(['index'],inplace=True,axis=1)
        dataset_x_ts.drop(['index'],inplace=True,axis=1)
        y=pd.get_dummies(dataset_y_tr.squeeze())
        y=y.values[:10].T
        X=dataset_x_tr.values.reshape(60000,784)[:10].T
        print('DataSet has been split into train and Validation set! 10% of data will be used as Validation Set')
        self.X = X
        self.y = y


    @classmethod
    def tearDownClass(self):
        print("\nClearing all the memories of the variables that are assigned in the Test_MLP class")
        del self.X
        del self.y
        gc.collect()

    def setUp(self):
        print("\nAdding layers and compiling before every test case")
        model = CodeFile.TLP(0.001)
        model.add(CodeFile.LayerRelu(5))
        model.add(CodeFile.LayerSigmoid(10))
        model.fit(self.X, self.y)
        model.compile(epochs=1)
        self.model = model

    def tearDown(self):
        print("Clearing the memories assigned to the variables in a test case")
        del self.model
        gc.collect()

    def test_TLP(self):
        # checking if alpha is initialised correctly or not
        self.assertEqual(self.model.getAlpha(),0.001)

        # checking the number of layers added
        self.assertEqual(len(self.model.getLayers()), 2)

        # checking the forwardPropagation functionality
        inpt = self.X[:, 0].reshape(-1, 1)
        a = CodeFile.TLP.forwardPropagation(self.model, inpt)
        flag = 1
        if (a.shape[0] != 10) or (
                a.shape[1] != 1):
            flag = 0
        self.assertEqual(flag, 1)

        # checking the backwardPropogation functinality
        y_cap = self.y[:, 0].reshape(-1, 1)
        lossDerivative = 2 * np.subtract(a, y_cap)
        CodeFile.TLP.backwardPropagation(self.model, lossDerivative)

        print("All test cases of test_TLP are passed!!")


    def test_Layers(self):
        # checking initializeLayer functionality
        np.random.seed(35)
        inputdims = self.X.shape[0]
        weights = np.random.rand(5, inputdims) * 10 ** (-9)
        bias = np.zeros((5, 1))
        flag = 1
        if (weights.shape[0] != self.model.getLayers()[0].getWeights().shape[0]) or (
                weights.shape[1] != self.model.getLayers()[0].getWeights().shape[1]):
            flag = 0
        if (bias.shape[0] != self.model.getLayers()[0].getBias().shape[0]) or (
                bias.shape[1] != self.model.getLayers()[0].getBias().shape[1]):
            flag = 0
        self.assertEqual(flag, 1)

        # checking getNodes functionality
        self.assertEqual(self.model.getLayers()[0].getNodes(), 5)

        # checking the functionality of forwardStep
        inpt = self.X[:, 0].reshape(-1, 1)
        relulayer = CodeFile.LayerRelu(5)
        relulayer.setWeights(self.model.getLayers()[0].getWeights())
        relulayer.setBias(self.model.getLayers()[0].getBias())
        a1 = CodeFile.Layers.forwardStep(relulayer, inpt)
        z = np.add((self.model.getLayers()[0].getWeights() @ self.X[:, 0].reshape(-1, 1)), self.model.getLayers()[0].getBias())
        a2 = 1 / (1 + np.exp(-z))
        flag=1
        if (a1.shape[0] != a2.shape[0]) or (
                a1.shape[1] != a2.shape[1]):
            flag = 0
        self.assertEqual(flag, 1)

        # checking the functionality of backStep
        sigmoidlayer = CodeFile.LayerSigmoid(10)
        sigmoidlayer.setWeights(self.model.getLayers()[1].getWeights())
        sigmoidlayer.setBias(self.model.getLayers()[1].getBias())
        out = CodeFile.Layers.forwardStep(sigmoidlayer, a1)
        y_cap = self.y[:, 0].reshape(-1, 1)
        lossDerivative = 2 * np.subtract(out, y_cap)
        derivativeA = sigmoidlayer.backStep(lossDerivative, 0.001)
        flag = 1
        if (a1.shape[0] != derivativeA.shape[0]) or (
                a1.shape[1] != derivativeA.shape[1]):
            flag = 0
        self.assertEqual(flag, 1)

        print("All test cases of test_Layers are passed!!")


if __name__=="__main__":
    unittest.main()
import cProfile
import CodeFile
import Loading_Data
import numpy as np
from sklearn import model_selection
import pandas as pd
import pstats
from functools import wraps


# boilerplate for profiling
def profile(output_file=None, sort_by='cumulative', lines_to_print=None, strip_dirs=False):
    

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _output_file = output_file or func.__name__ + '.prof'
            pr = cProfile.Profile()
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()
            pr.dump_stats(_output_file)

            with open(_output_file, 'w') as f:
                ps = pstats.Stats(pr, stream=f)
                if strip_dirs:
                    ps.strip_dirs()
                if isinstance(sort_by, (tuple, list)):
                    ps.sort_stats(*sort_by)
                else:
                    ps.sort_stats(sort_by)
                ps.print_stats(lines_to_print)
            return retval
        return wrapper
    return inner

@profile(output_file='Profiling_Info.txt', sort_by='cumulative', strip_dirs=True)
def profiling_info(X, y):
    model = CodeFile.TLP(0.001)
    model = model
    model.add(CodeFile.LayerRelu(5))
    model.add(CodeFile.LayerSigmoid(10))
    model.fit(X, y)
    model.compile(epochs=1)


if __name__=="__main__":

    data=Loading_Data.DataFrameLoader()
    dataset_x_tr, dataset_x_ts,dataset_y_tr, dataset_y_ts=data.load_dataframes()
    dataset_y_tr.drop(['index'],inplace=True,axis=1)
    dataset_x_tr.drop(['index'],inplace=True,axis=1)
    dataset_y_ts.drop(['index'],inplace=True,axis=1)
    dataset_x_ts.drop(['index'],inplace=True,axis=1)
    y=pd.get_dummies(dataset_y_tr.squeeze())
    y=y.values.T
    X=dataset_x_tr.values.reshape(60000,784).T
    # Features
    
    print('DataSet has been split into train and Validation set! 10% of data will be used as Validation Set')


    profiling_info(X, y)


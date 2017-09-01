__author__  = "atadrous"
__version__ = 0.0


#Start
class Main:
    def __init__(self):
        self.var_1=0

    def Start(self):
        print "...."
        # Python version
        import sys
        print('Python: {}'.format(sys.version))
        # scipy
        import scipy
        print('scipy: {}'.format(scipy.__version__))
        # numpy
        import numpy
        print('numpy: {}'.format(numpy.__version__))
        # matplotlib
        import matplotlib
        print('matplotlib: {}'.format(matplotlib.__version__))
        # pandas
        import pandas
        print('pandas: {}'.format(pandas.__version__))
        # statsmodels
        import statsmodels
        print('statsmodels: %s' % statsmodels.__version__)
        # scikit-learn
        import sklearn
        print('sklearn: {}'.format(sklearn.__version__))

    def HelloWorld(self):
        # Load libraries
        import pandas
        from pandas.plotting import scatter_matrix
        import matplotlib.pyplot as plt
        from sklearn import model_selection
        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.naive_bayes import GaussianNB
        from sklearn.svm import SVC

        # Load dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        dataset = pandas.read_csv(url, names=names)

        print(dataset.shape)
        print(dataset.head(5))
        print(dataset.describe())
        print(dataset.groupby('class').size())

        #dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
        #plt.show()

        #dataset.hist()
        #plt.show()

        # scatter plot matrix
        scatter_matrix(dataset)
        plt.show()

        # Split-out validation dataset
        array = dataset.values
        X = array[:, 0:4]
        Y = array[:, 4]
        validation_size = 0.20
        seed = 7
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                    random_state=seed)



    def test(self):
        #dataframe
        import numpy as np
        import pandas

        myarray = np.array([[1, 2, 3], [4, 5, 6]])
        rownames = ['a', 'b']
        colnames = ['one', 'two', 'three']
        mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
        print(mydataframe)

        a= np.arange(4)
        print a

        #x = np.array([1.6, 2 , 0.1 , -1], dtype=np.int64)  # Force a particular datatype
        #print x
        #print(x.dtype)  # Prints "int64"
        #print(np.dot(a, x)) # matrix multiplication
        #print mydataframe.T #Transport


        aa = np.tile(a, (4, 1))  # Stack 4 copies of v on top of each other
        print aa

        v = np.array([1, 2, 3])  # v has shape (3,)
        print(np.reshape(v, (3, 1)))

        x = np.arange(0, 3 * np.pi, 0.1)
        print x


        return

#Main().Start()
#Main().HelloWorld()
Main().test()


import numpy as np

class KNN():

    def __init__(self, model_name):
        self.model_name = model_name
        self.dataset = None

    def set_dataset(self, data, numpy_array=True):
        if not numpy_array:
            self.dataset = np.asarray(data)
        else:
            self.dataset = data

    def predict(self, data, append=False):
        for row in data:
            pass 

    def cal_dist(self, x, x_o):
        # by default use euclidean distance
        assert x.shape[0] == 1
        assert x.shape[1] == x_o.shape[1]
        dist = 0
        for i in x.shape[1]:
            dist = (x[0][i] - x_o[0][i]) ** 2
        return dist ** (1/2)


def run_KNN_tests():
    
    # Test the initialization of the model
    model = KNN(model_name="Basic Test Model")
    assert model.model_name == "Basic Test Model"

    # Test Dataset setting with lists
    data = [[1,2,3], [4,5,6]]
    model.set_dataset(data, numpy_array=False)
    assert str(type(model.dataset)) == "<class 'numpy.ndarray'>"

    # Test Dataset setting with numpy array
    dataset = np.asarray(data)
    assert np.array_equal(model.dataset, dataset)
    model.set_dataset(dataset)
    assert str(type(model.dataset)) == "<class 'numpy.ndarray'>"

    # Completed all tests
    print("KNN TESTING COMPLETED!")


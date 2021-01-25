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
        if str(type(x_o)) == "<class 'numpy.ndarray'>":
            x_o = np.asarray(x_o)
        dist = 0
        for i in range(x.shape[0]):
            dist += (x[i] - x_o[i]) ** 2
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

    # Testing Distance Calculation
    new_data_point = np.asarray([1,2,3])
    assert model.cal_dist(model.dataset[0], new_data_point) == 0

    # Completed all tests
    print("KNN TESTING COMPLETED!")


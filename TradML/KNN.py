import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class KNN():

    def __init__(self, model_name, p_value=2, k_value=3, model_type='classification'):
        self.model_name = model_name
        self.dataset = None
        self.p_value = p_value
        self.k_value = k_value
        self.model_type = model_type

    def set_dataset(self, data, numpy_array=True):
        if not numpy_array:
            self.dataset = np.asarray(data)
        else:
            self.dataset = data

    def predict(self, data, append=False, extra_stats=False):
        distances = dict()
        for index, row in enumerate(self.dataset):
            distances[index] = self.cal_dist(row[:-1], data)
        final_distances = dict(sorted(distances.items(), key=lambda item: item[1]))
        if self.model_type == "classification":
            k_classes = self.get_classes(list(final_distances.keys())[:self.k_value])
            return k_classes, max(k_classes, key=k_classes.count)
        else:
            k_values = self.get_classes(list(final_distances.keys())[:self.k_value])
            return k_values, sum(k_values) / len(k_values)

    def get_classes(self, data):
        assert len(data) == self.k_value
        return [self.dataset[row,-1] for row in data]

    def cal_dist(self, x, x_o):
        # by default use euclidean distance
        if str(type(x_o)) != "<class 'numpy.ndarray'>":
            x_o = np.asarray(x_o)
        dist = 0
        for i in range(x.shape[0]):
            dist += (abs(x[i] - x_o[i])) ** self.p_value
        return dist ** (1/self.p_value)


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

    # create dataset
    dataset = np.asarray([[0,1,0],[0,.8,0],[0,0.6,0],[.4,.4,0],[1,0,1],[.8,0,1],[.6,.2,1],[.7,.2,1]])
    model.dataset = dataset
    print(model.predict([.5, .1]))
    print(model.predict([.3, .6]))

    # Completed all tests
    print("KNN TESTING COMPLETED!")


import numpy as np
class Knn:
    """
    Class to store data for regression problems
    """

    def __init__(self, x_train, y_train, K=5):
        """
        Creates a kNN instance

        :param x_train: numpy array with shape (n_rows,1)- e.g. [[1,2],[3,4]]
        :param y_train: numpy array with shape (n_rows,)- e.g. [1,-1]
        :param K: The number of nearest points to consider in classification
        """

        # Import and build the BallTree on training features
        from sklearn.neighbors import BallTree
        self.balltree = BallTree(x_train)

        # Cache training labels and parameter K
        self.y_train = y_train
        self.K = K

    def majority(self, neighbor_indices, neighbor_distances=None):
        """
        Given indices of nearest neighbors in training set, return the majority label.
        Break ties by considering 1 fewer neighbor until a clear winner is found.

        :param neighbor_indices: The indices of the K nearest neighbors in self.X_train
        :param neighbor_distances: Corresponding distances from query point to K nearest neighbors.
        """
        lable = set(self.y_train)
        lable_count = {i: 0 for i in lable}
        k = self.K
        temp_max = {'number': None, 'count': 0, 'is_repeated': False}
        far_point = {'index': None, 'distance': 0}
        # your code here
        while True:
            for i in range(k):
                index = neighbor_indices[i]
                distance = neighbor_distances[i]
                lable_count[self.y_train[index]] += 1
                if far_point['distance'] <= distance:
                    far_point['distance'] = distance
                    far_point['index'] = index

            for i in lable:
                if temp_max['count'] < lable_count[i]:
                    temp_max['number'] = i
                    temp_max['count'] = lable_count[i]
                    temp_max['is_repeated'] = False
                elif temp_max['count'] == lable_count[i]:
                    temp_max['is_repeated'] = True

            if temp_max['is_repeated']:
                k -= 1
                lable_count = {i: 0 for i in set(self.y_train)}
                temp_max = {'number': None, 'count': 0, 'is_repeated': False}
                far_point = {'index': None, 'distance': 0}
            else:
                break

        return temp_max['number']

    def classify(self, x):
        """
        Given a query point, return the predicted label

        :param x: a query point stored as an ndarray
        """
        # your code here

        distances, indexes = self.balltree.query([x], self.K)
        return self.majority(indexes[0], neighbor_distances=distances[0])

    def predict(self, X):
        """
        Given an ndarray of query points, return yhat, an ndarray of predictions

        :param X: an (m x p) dimension ndarray of points to predict labels for
        """
        # your code here
        yhat = []
        for i in range(X.shape[0]):
            yhat.append(self.classify(X[i]))
        return np.array(yhat)

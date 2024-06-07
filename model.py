import numpy as np
from matplotlib import pyplot as plt
from KNN import Knn
from MNIST_import import MNIST
from extract_num import ExtractNum
import tensorflow as tf
import os


class Task:
    def __init__(self, image_path='sample/', temp_path='temp/', target_path='result/result.xlsx', mode='cnn'):
        self.image_path = image_path
        self.temp_path = temp_path
        self.target_path = target_path

        self.tasks = self.handle_path()

        self.data = MNIST()
        self.model_knn = Knn(self.data.train_x, self.data.train_y, K=3)
        self.model_cnn = tf.keras.models.load_model('temp/my_model2.h5')
        self.mode = mode
        self.result = []

    def run(self):
        if self.mode == 'cnn':
            self.result = self.cnn_run()
        elif self.mode == 'knn':
            self.result = self.knn_run()

    def handle_path(self):
        tasks = []
        directory_path = self.image_path
        i = 1
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    temp = ExtractNum(file_path, self.temp_path + '/' + file, self.target_path, i)
                    tasks.append(temp)
                    i += 1
        return tasks

    def knn_run(self):
        res = []
        for task in self.tasks:
            sample = task.normalized_num
            predictions = self.model_knn.predict(sample)
            temp = task.convert2nums(predictions)
            task.excel_writer(temp)
            res.append(temp)
        return res

    def cnn_run(self):
        res = []
        for task in self.tasks:
            temp = task.mnist_digits
            sample = np.array(temp).reshape(len(temp), 28, 28, 1)
            predictions = self.model_cnn.predict(sample)
            temp = task.convert2nums([prediction.argmax() for prediction in predictions])
            task.excel_writer(temp)
            res.append(temp)
        return res

    def view_digit(self, x, label=True):
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if label:
            plt.xlabel("true: {}".format(label), fontsize=16)
        plt.show()

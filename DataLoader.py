import cv2
import numpy as np
from os.path import join, exists
from os import listdir, walk


class DataLoader:
    def __init__(self, train_path="data/train", test_path="data/test"):
        self.train_path = train_path
        self.test_path = test_path

    def load_train_data(self):
        X_train, y_train = [], []
        if exists("data/x_train.npy") and exists("data/y_train.npy"):
            X_train = np.load("data/x_train.npy", allow_pickle=True)
            y_train = np.load("data/y_train.npy", allow_pickle=True)
        else:
            for i in listdir(self.train_path):
                h, w, c = None, None, None
                for image in listdir(join(self.train_path, i, 'images')):
                    image_ = cv2.imread(join(self.train_path, i, 'images', image), 1)
                    image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
                    h, w, c = image_.shape
                    X_train.append(image_)

                mask = np.zeros((h, w, 1))
                for mask_ in listdir(join(self.train_path, i, "masks")):
                    current = cv2.imread(join(self.train_path, i, 'masks', mask_), -1)
                    current = np.expand_dims(current, axis=-1)
                    mask = np.maximum(mask, current)
                y_train.append(mask)
            X_train, y_train = np.array(X_train), np.array(y_train)
            np.save("data/x_train.npy", X_train)
            np.save("data/y_train.npy", y_train)

        return X_train, y_train

    def load_test_data(self, load_masks):
        if exists("data/x_test.npy") and exists("data/y_test.npy") and load_masks:
            X_test = np.load("data/x_test.npy", allow_pickle=True)
            y_test = np.load("data/y_test.npy", allow_pickle=True)

            return X_test, y_test

        elif exists("data/x_test.npy"):
            X_test = np.load("data/x_test.npy", allow_pickle=True)
            return X_test
        else:
            X_test = []
            for i in listdir(self.test_path):
                for image in listdir(join(self.test_path, i, 'images')):
                    image_ = cv2.imread(join(self.test_path, i, 'images', image))
                    X_test.append(image_)
            np.save("data/x_test.npy", X_test)
            return np.array(X_test)

    def load_labels(self, path):
        indexes = next(walk(path))[1]

        return indexes


if __name__ == "__main__":
    data = DataLoader()
    data.load_labels("data/stage_2")

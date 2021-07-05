
from Input.process.solve import get_shape
import numpy as np

# load data from file .txt
X_train = np.loadtxt("F:/NhanDienLogo/Input/Files/train_X.txt").reshape(get_shape()[0])
y_train = np.loadtxt("F:/NhanDienLogo/Input/Files/train_y.txt").reshape(get_shape()[1])

# return data training
def _data():
    return [X_train, y_train]

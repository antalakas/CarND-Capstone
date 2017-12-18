from __future__ import print_function
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from model import Model

data = pickle.load(open('dataset_tl_py2.pickle', 'rb'))
X_train = data[0]
y_train = data[1]

X_train, y_train = shuffle(X_train, y_train, random_state=143)

num_classes = 3
class_name = ["RED", "GREEN", "UNKNOWN"]

X_train, X_dev, y_train, y_dev = train_test_split(
    X_train, y_train,
    test_size=0.10, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train,
    test_size=0.05, random_state=42
)

print(X_train.shape)
print(X_dev.shape)
print(X_test.shape)

EPOCHS = 100
BATCH_SIZE = 100

def batchify(X_train, y_train, batch_size):
    for offset in range(0, X_train.shape[0], BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_train[offset:end], y_train[offset:end]
        yield batch_x, batch_y

model = Model()

best_dev_acc = 0

print("Training...")
for epoch in xrange(EPOCHS):
    X_train, y_train = shuffle(X_train, y_train)
    for batch_x, batch_y in batchify(X_train, y_train, BATCH_SIZE):
        model.train(batch_x, batch_y)
    train_accuracy = 0
    total = 0
    for batch_x, batch_y in batchify(X_train, y_train, BATCH_SIZE):
        train_accuracy += model.evaluate(batch_x, batch_y) * batch_x.shape[0]
        total += batch_x.shape[0]
    train_accuracy = train_accuracy / total
    dev_accuracy = model.evaluate(X_dev, y_dev)
    print("Epoch {}... Train Acc {:.4f} Dev Acc {:.4f}...".format(
            epoch + 1,
            train_accuracy,
            dev_accuracy
        )
        , end=' '
    )
    if dev_accuracy > best_dev_acc:
        best_dev_acc = dev_accuracy
        model.save("best_dev.ckpt")
        print("Best Dev beaten", end=' ')
    print()

model.load("best_dev.ckpt")
print("{}".format(model.evaluate(X_dev, y_dev)))
print("Test Accuracy {:.4f}".format(model.evaluate(X_test, y_test)))

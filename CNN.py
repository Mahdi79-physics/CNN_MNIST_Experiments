import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Define a function to create the CNN model
def create_model(use_batch_norm=False):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu')
    ])

    if use_batch_norm:
        # Add batch normalization layers
        model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Create and train the model with batch normalization
model_with_bn = create_model(use_batch_norm=True)
history_with_bn = model_with_bn.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Create and train the model without batch normalization
model_without_bn = create_model(use_batch_norm=False)
history_without_bn = model_without_bn.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Plotting the results
plt.figure(figsize=(12, 5))

# Accuracy comparison
plt.subplot(1, 2, 1)
plt.plot(history_with_bn.history['accuracy'], label='With Batch Norm - Training')
plt.plot(history_with_bn.history['val_accuracy'], label='With Batch Norm - Validation')
plt.plot(history_without_bn.history['accuracy'], label='Without Batch Norm - Training')
plt.plot(history_without_bn.history['val_accuracy'], label='Without Batch Norm - Validation')
plt.title('Training and Validation Accuracy')
plt.legend()

# Loss comparison
plt.subplot(1, 2, 2)
plt.plot(history_with_bn.history['loss'], label='With Batch Norm - Training')
plt.plot(history_with_bn.history['val_loss'], label='With Batch Norm - Validation')
plt.plot(history_without_bn.history['loss'], label='Without Batch Norm - Training')
plt.plot(history_without_bn.history['val_loss'], label='Without Batch Norm - Validation')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()




import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize to range 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the input data
x_train = x_train.reshape((x_train.shape[0], -1))  # Reshape input to be one-dimensional
x_test = x_test.reshape((x_test.shape[0], -1))  # Reshape input to be one-dimensional

model1 = Sequential()
model1.add(Dense(512, activation='relu', input_shape=(784,)))
model1.add(Dense(512, activation='relu'))
model1.add(Dense(10, activation='softmax'))
# Define MLP model
model2 = Sequential()
model2.add(Dense(512, activation='relu', input_shape=(784,)))
model2.add(Dropout(0.2))
model2.add(Dense(512, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(10, activation='softmax'))

# Compile the model
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history1 = model1.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))
history2 = model2.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))


# Evaluate on test data
test_loss, test_acc = model1.evaluate(x_test, y_test)
print("Test accuracy: ", test_acc)
test_loss, test_acc = model2.evaluate(x_test, y_test)
print("Test accuracy: ", test_acc)



import matplotlib.pyplot as plt

# Plotting the results
plt.figure(figsize=(12, 5))

# Accuracy comparison
plt.subplot(1, 2, 1)
plt.plot(history1.history['accuracy'], label='Without Dropout - Training')
plt.plot(history1.history['val_accuracy'], label='Without Dropout - Validation')
plt.plot(history2.history['accuracy'], label='With Dropout - Training')
plt.plot(history2.history['val_accuracy'], label='With Dropout - Validation')
plt.title('Training and Validation Accuracy')
plt.legend()

# Loss comparison
plt.subplot(1, 2, 2)
plt.plot(history1.history['loss'], label='Without Dropout - Training')
plt.plot(history1.history['val_loss'], label='Without Dropout - Validation')
plt.plot(history2.history['loss'], label='With Dropout - Training')
plt.plot(history2.history['val_loss'], label='With Dropout - Validation')
plt.title('Training and Validation Loss')
plt.legend()



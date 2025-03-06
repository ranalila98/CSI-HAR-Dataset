from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import os
from tensorflow.keras.layers import LeakyReLU

# Define dataset path
base_path = "E:\CSI-HAR\data\generated images"  # Update if needed
train_path = os.path.join(base_path, "Train")
test_path = os.path.join(base_path, "Test")

# Initialize the CNN
model = Sequential()

# Convolution and Pooling layers
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
model.add(Flatten())

# Fully Connected Layers
model.add(Dense(128, activation='linear'))
model.add(Dropout(0.1))
model.add(Dense(7, activation='softmax'))

# Compile Model
opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation
train_datagen = ImageDataGenerator(featurewise_center=False)
test_datagen = ImageDataGenerator(featurewise_center=False)

# Load Data
target_size = (64, 64)
classes = ['lie down','fall','bend', 'run', 'sitdown','standup','walk']
train_set = train_datagen.flow_from_directory(train_path, target_size=target_size, class_mode="categorical", batch_size=32, classes=classes)
test_set = test_datagen.flow_from_directory(test_path, target_size=target_size, class_mode="categorical", batch_size=32, classes=classes)

# Checkpoint to save the best model
checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Train Model
history = model.fit(train_set, epochs=150, validation_data=test_set, callbacks=[checkpoint])

# Save Model
model_json = model.to_json()
with open("Classifier.json", "w") as json_file:
    json_file.write(model_json)

# Plot Accuracy and Loss
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')
plt.show()

# Confusion Matrix
def plot_confusion_matrix(cm, classes):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]:.2f}", horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2. else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

predictions = model.predict(test_set)
y_pred = np.argmax(predictions, axis=1)
y_true = test_set.classes
cnf_matrix = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cnf_matrix, classes)

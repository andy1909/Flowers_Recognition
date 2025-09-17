import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import EarlyStopping

# Dataset paths
PATH_DAISY = '/home/long/Documents/AI/Nhan_Dien_Hoa/archive/train/daisy'          
PATH_DANDELION = '/home/long/Documents/AI/Nhan_Dien_Hoa/archive/train/dandelion'
PATH_ROSES = '/home/long/Documents/AI/Nhan_Dien_Hoa/archive/train/rose'
PATH_SUNFLOWERS = '/home/long/Documents/AI/Nhan_Dien_Hoa/archive/train/sunflower'
PATH_TULIPS = '/home/long/Documents/AI/Nhan_Dien_Hoa/archive/train/tulip'

MODEL_FILE = 'flower_ann_model_improved.h5' 

# -------------------------------------------------------------------------
# HYPERPARAMETERS
# -------------------------------------------------------------------------
IMG_HEIGHT = 100
IMG_WIDTH = 100
NUM_CLASSES = 5
BATCH_SIZE = 1024
EPOCHS = 100 
EARLY_STOPPING_PATIENCE = 15
#---------------------------------------------------------------------------

print("Starting data loading and processing...")

images = []
labels = []

path_to_label = {
    PATH_DAISY: 0,
    PATH_DANDELION: 1,
    PATH_ROSES: 2,
    PATH_SUNFLOWERS: 3,
    PATH_TULIPS: 4
}
CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

for path, label_idx in path_to_label.items():
    class_name = CLASS_NAMES[label_idx]
    print(f"Reading data for class '{class_name}' from: {path}")
    
    if not os.path.isdir(path):
        print(f"!!! Warning: Path does not exist, skipping: {path}")
        continue
   
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)           
            images.append(img_rgb)
            labels.append(label_idx)

print(f"\nSuccessfully loaded and processed {len(images)} images.")

images = np.array(images, dtype='float32') / 255.0
labels = np.array(labels)

y_data = to_categorical(labels, num_classes=NUM_CLASSES)

x_train, x_test, y_train, y_test = train_test_split(
    images, y_data, test_size=0.2, random_state=42, stratify=y_data
)

print(f"Training set size: {x_train.shape}")
print(f"Test set size: {x_test.shape}")


print("Building a simple ANN model architecture...")
model = Sequential([
    Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopper = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, verbose=1, restore_best_weights=True)

print("\nStarting the training process...")
history = model.fit(x_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stopper])
print("Training finished!")

model.save(MODEL_FILE)
print(f"\nModel successfully saved to file: '{MODEL_FILE}'")


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
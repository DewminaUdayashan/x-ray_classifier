import tensorflow as tf
import matplotlib.pyplot as plt

# 1. CONFIGURATION
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = 'data/main/train'
EPOCHS = 15

# 2. LOAD DATASET
print("Loading training and validation data...")
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='categorical', # Use 'categorical' for 3+ classes
    image_size=IMG_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    seed=123,
    validation_split=0.2,
    subset='both'
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print(f"\nFound {NUM_CLASSES} classes: {class_names}")

# 3. BUILD THE MODEL
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    # The output layer must have NUM_CLASSES neurons and 'softmax'
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# 4. COMPILE THE MODEL
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', # Loss function for multi-class
    metrics=['accuracy']
)
model.summary()

# 5. TRAIN THE MODEL
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# 6. SAVE & VISUALIZE
model.save('saved_models/main_image_classifier.keras')
print("\nModel saved as 'main_image_classifier.keras'")

# 9. VISUALIZE TRAINING RESULTS

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
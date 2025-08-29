import tensorflow as tf
import matplotlib.pyplot as plt

# 1. CONFIGURATION
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

DATA_DIR = 'data/train/scoliosis'
EPOCHS = 15

# 2. LOAD DATASET
print("Loading training and validation data...")
# The model will learn the classes 'c_type' and 's_type' from the folder names
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='binary', # Use 'binary' for two classes
    image_size=IMG_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    seed=123,
    validation_split=0.2,
    subset='both'
)

class_names = train_ds.class_names
print(f"\nFound classes: {class_names}") # Should be ['c_type', 's_type']

# 3. BUILD & COMPILE MODEL
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# 4. TRAIN THE MODEL
print("\nStarting model training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# 5. SAVE THE MODEL
model.save('saved_models/scoliosis_curve_classifier.keras')
print("\nModel saved as 'scoliosis_curve_classifier.keras'")


# --- 6. VISUALIZE TRAINING RESULTS ---
print("Generating training history plot...")

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
plt.ylabel('Cross Entropy Loss')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
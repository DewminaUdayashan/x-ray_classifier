import tensorflow as tf
import matplotlib.pyplot as plt
import os

# 1. SETUP AND CONFIGURATION

# Define key parameters
IMG_SIZE = (224, 224) # Input image size for the model
BATCH_SIZE = 32      # Number of images to process in a batch
DATA_DIR = 'data'    # Path to your main data folder
EPOCHS = 15          # Number of times to train on the entire dataset

# 2. LOAD AND PREPARE THE DATASET

# Use Keras utility to load images from directories.
# It automatically infers labels ('normal', 'scoliosis') from the folder names.
# We'll also split the training data into training (80%) and validation (20%).

print("Loading training and validation data...")
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'train'),
    labels='inferred',
    label_mode='binary', # Use 'binary' for two classes
    image_size=IMG_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    seed=123,
    validation_split=0.2,
    subset='both'
)

print("\nLoading test data...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'test'),
    labels='inferred',
    label_mode='binary',
    image_size=IMG_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=False # No need to shuffle test data
)

# Get the class names
class_names = train_ds.class_names
print(f"\nClass names found: {class_names}") # Should be ['normal', 'scoliosis']

# 3. DATA AUGMENTATION AND PREPROCESSING

# Create a data augmentation layer. This helps the model generalize better
# by creating modified versions of images during training.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Rescaling layer to normalize pixel values from [0, 255] to [0, 1]
rescale = tf.keras.layers.Rescaling(1./255)

# 4. BUILD THE MODEL USING TRANSFER LEARNING

# Load a pre-trained model (MobileNetV2) without its final classification layer.
# We'll use its learned features and add our own classifier on top.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False, # turning off final ImageNet classifier
    weights='imagenet'
)

# Freeze the base model's layers so we don't change its learned weights during initial training.
base_model.trainable = False

# Create the final model by stacking our layers on top of the base model.
inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)  # Apply augmentation first
x = rescale(x)                 # Then rescale
x = base_model(x, training=False) # Run the base model in inference mode
x = tf.keras.layers.GlobalAveragePooling2D()(x) # Pool the features
x = tf.keras.layers.Dropout(0.2)(x)             # Add dropout for regularization
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x) # Final output neuron for binary classification

model = tf.keras.Model(inputs, outputs)

# 5. COMPILE THE MODEL

# Configure the model for training.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy', # Perfect loss function for two classes
    metrics=['accuracy']
)

# Print a summary of the model architecture
model.summary()

# 6. TRAIN THE MODEL

print("\nStarting model training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)
print("Training finished!")

# 7. EVALUATE THE MODEL

print("\nEvaluating model on the test dataset...")
loss, accuracy = model.evaluate(test_ds)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# 8. SAVE THE MODEL

model.save('saved_models/scoliosis_classifier.keras')
print("\nModel saved as 'scoliosis_classifier.keras'")

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
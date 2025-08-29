import tensorflow as tf

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
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# 5. SAVE THE MODEL
model.save('saved_models/scoliosis_curve_classifier.keras')
print("\nModel saved as 'scoliosis_curve_classifier.keras'")
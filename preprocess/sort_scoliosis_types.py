import os
import shutil

data_paths = [
    'data/train/scoliosis',
    'data/test/scoliosis'
]

print("Starting to sort files...")

for path in data_paths:
    if not os.path.exists(path):
        print(f"Warning: Path not found, skipping: {path}")
        continue

    # Define the destination folders
    s_type_path = os.path.join(path, 's_type')
    c_type_path = os.path.join(path, 'c_type')

    # Create the folders if they don't exist
    os.makedirs(s_type_path, exist_ok=True)
    os.makedirs(c_type_path, exist_ok=True)

    # List all files in the directory
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    for filename in files:
        # Check if the second part of the filename is 'S'
        parts = filename.split(',')
        if len(parts) > 1 and parts[1].strip().upper() == 'S':
            # This is an S-Type
            shutil.move(os.path.join(path, filename), os.path.join(s_type_path, filename))
            print(f"Moved {filename} to s_type")
        else:
            # Assume everything else is a C-Type
            shutil.move(os.path.join(path, filename), os.path.join(c_type_path, filename))
            print(f"Moved {filename} to c_type")

print("\nFile sorting complete!")

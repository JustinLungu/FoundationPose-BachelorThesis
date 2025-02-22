from hots import load_HOTS_scenes  # Import the dataset loader

# Load dataset
train_data, test_data = load_HOTS_scenes(root='HOTS_v1', transform=True)

# Example: Access an image and its target (bounding boxes, masks, etc.)
img, target = train_data[0]

print(img)
print(target)

# Then you can integrate this data into your model’s training or testing pipeline


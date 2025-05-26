import os
from collections import Counter
import matplotlib.pyplot as plt
import yaml

# Paths
base_path = 'dataset'
splits = ['train', 'valid', 'test']

# Load class names from data.yaml
with open(os.path.join(base_path, 'data.yaml'), 'r') as file:
    data = yaml.safe_load(file)
    class_names = data['names']

# Count class instances from all label files
class_counts = Counter()

for split in splits:
    label_dir = os.path.join(base_path, split, 'labels')
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(label_dir, filename), 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_names[class_id]] += 1

# Display counts
print("Class Distribution:", class_counts)

# Pie chart
plt.figure(figsize=(6, 6))
plt.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%', startangle=90)
plt.title('Class Distribution of Milkfish Dataset')
plt.show()

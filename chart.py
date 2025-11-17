import os
from collections import Counter
import matplotlib.pyplot as plt
import yaml
import numpy as np

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
    if not os.path.isdir(label_dir):
        continue
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(label_dir, filename), 'r') as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    class_id = int(parts[0])
                    class_counts[class_names[class_id]] += 1

# Prepare ordered labels and counts (preserve order from data.yaml)
labels = [class_names[i] for i in range(len(class_names))]
counts = [class_counts.get(lbl, 0) for lbl in labels]
total = sum(counts)

# Display counts
print("Class Distribution:", dict(zip(labels, counts)))

if total == 0:
    print("No labels found. Check dataset paths.")
else:
    # Grayscale colors
    cmap = plt.get_cmap('gray')
    if len(labels) > 1:
        colors = [cmap(0.2 + 0.6 * i / (len(labels) - 1)) for i in range(len(labels))]
    else:
        colors = [cmap(0.5)]

    # Pie chart with counts shown in each wedge
    def autopct_with_count(pct):
        absolute = int(round(pct / 100.0 * total))
        return f"{pct:.1f}%\n({absolute})"

    plt.figure(figsize=(6, 6))
    wedges, texts, autotexts = plt.pie(
        counts,
        labels=labels,
        autopct=autopct_with_count,
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'white'}
    )
    plt.setp(autotexts, size=9, weight="bold", color="black")
    plt.title('Class Distribution of Milkfish Dataset (counts & %)', color='black')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

    # Optional: bar chart with counts and annotations (grayscale)
    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, counts, color=colors, edgecolor='black')
    plt.ylabel('Count')
    plt.title('Class Counts', color='black')
    plt.xticks(rotation=45, ha='right')
    for bar, cnt in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(1, total*0.01), str(cnt),
                 ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()

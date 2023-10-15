import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load pickled data
with open('data.pkl', 'rb') as file:
    data = pickle.load(file)

labels = np.array(data['labels']) # greyscale where pixel values 0, 1, 2, 3 correspond to background, muscle, fat, bone

n = len(np.unique(labels[0]))

counts_list = [[] for _ in range(n)]
for i in range(len(labels)):
    unique, counts = np.unique(labels[i], return_counts=True)
    for j in range(n):
        counts_list[j].append(counts[j])

average = [np.mean(counts_list[i])/(256**2) for i in range(n)]

# Data to plot
labels = 'Background', 'Muscle', 'Fat', 'Bone'
colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e']

# Plot
plt.pie(average, labels=labels, colors=colors, autopct='%1.1f%%', startangle=10)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Average class weights of labels')
plt.show()

inverse_average = [1/average[i] for i in range(n)]
print(inverse_average)
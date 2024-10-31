import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'df_filtered.csv'
df = pd.read_csv(file_path)

# Plot the histograms for Total Cash Compensation and Overtime
plt.figure(figsize=(12, 6))

# Histogram for Total Cash Compensation
plt.subplot(1, 2, 1)
plt.hist(df['Total Cash Compensation'], bins=30, color='blue', alpha=0.7, edgecolor='black')
plt.title('Distribution of Total Cash Compensation')
plt.xlabel('Total Cash Compensation')
plt.ylabel('Frequency')

# Histogram for Overtime
plt.subplot(1, 2, 2)
plt.hist(df['Overtime'], bins=30, color='green', alpha=0.7, edgecolor='black')
plt.title('Distribution of Overtime')
plt.xlabel('Overtime Pay')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Scatterplot for Overtime vs. Total Compensation
plt.figure(figsize=(8, 6))
plt.scatter(df['Overtime'], df['Total Cash Compensation'], alpha=0.5, color='red')
plt.title('Overtime vs. Total Compensation')
plt.xlabel('Overtime Pay')
plt.ylabel('Total Compensation')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#3D audio data, 45296 rows, 175 columns and 44 depth.
#45296 rows = 45296 sound samples.
#175 columns = 175 features.
#44 depth = 44 time steps.
#The features are obtained through FFT and MFCC. Their semantic meanings are given in the idx_to_feature_name.csv file.
data = np.load('development.npy')

#every word here corresponds to every row(sound sample) in the data. 
#It goes in depth as well, so the first word corresponds to the first row, the second word to the second row, etc.
labels_df = pd.read_csv('development.csv')
labels = labels_df['word'].tolist() 


feature_df = pd.read_csv('idx_to_feature_name.csv')
feature_names = feature_df['feature_name'].tolist()

#A list containing numbers from 1 to 44, representing the time steps.
time_steps = list(range(1, 45))

print("Shape of data is: ", data.shape)

print("Labels are listed as: ", labels)


#Visualizing distributions of each feature in the dataset.
fig, axes = plt.subplots(59, 3, figsize=(20, 100))
axes = axes.flatten()
for i in range(175):
    axes[i].hist(data[:, i, :].flatten(), bins=50, color='blue', alpha=0.7, rwidth=0.85)
    axes[i].set_title(feature_names[i])
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

#In order to obtain correlation between features, timestamp 0 is selected for each sample and all features.
first_timestamp_features = data[:, :, 0]
features_df = pd.DataFrame(first_timestamp_features, columns=feature_names)
#Optionally, we can also join this df with the labels if needed for analysis.
#features_df = features_df.join(labels_df['word'])
print("Features and their respective values for every sample at timestamp 0 is: ", features_df)
#Now we have a 2D features_df DataFrame with samples as rows, features as columns.

#Calculating the correlation matrix for the features.
correlation_matrix = features_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.show()
#Both from visual inspection and the correlation matrix, we can see that the features from same "family" are highly correlated,
#therefore use of all of them is redundant.

#Plotting histograms for the first timestamp of each feature for 3 speakers.
speaker_ids = labels_df['speaker_id'].values
speaker1_data = data[np.where(speaker_ids == 1)]
speaker2_data = data[np.where(speaker_ids == 50)]
speaker3_data = data[np.where(speaker_ids == 3)]
fig, axes = plt.subplots(59, 3, figsize=(20, 100))  
axes = axes.flatten()
#Iterate through each feature and plot histograms for speakers
for i in range(175):  
    #Flatten the data for the i-th feature across all timestamps for speaker 1 and speaker 2
    speaker1_feature_data = speaker1_data[:, i, :].flatten()
    speaker2_feature_data = speaker2_data[:, i, :].flatten()
    speaker3_feature_data = speaker3_data[:, i, :].flatten()
 
    axes[i].hist(speaker1_feature_data, bins=50, alpha=0.5, label='Speaker 1')
    axes[i].hist(speaker2_feature_data, bins=50, alpha=0.5, label='Speaker 2')
    axes[i].hist(speaker3_feature_data, bins=50, alpha=0.5, label='Speaker 3')
    axes[i].set_title(feature_names[i])
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    axes[i].legend(loc='upper right')
plt.tight_layout()
plt.show()






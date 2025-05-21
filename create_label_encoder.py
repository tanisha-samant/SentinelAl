from sklearn.preprocessing import LabelEncoder
import pickle

# Define the audio classes
sound_classes = ['aggressive_speech', 'background_noise', 'normal_speech', 'scream']

# Create and fit the label encoder
label_encoder = LabelEncoder()
label_encoder.fit(sound_classes)

# Save the label encoder to a pickle file
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("label_encoder.pkl created successfully!")
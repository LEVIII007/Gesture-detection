# How to run this?
install requirement.txt file
run app.py: DTW version
run app1.py for the cosine similarity version
in line 101, you can change file path to desired gesture file
in line 105, you can change filepath to the desired test video in which you want to detect gesture.


# Gesture Detection in Video Sequences
This project aims to develop a prototype to detect a specific gesture within a video sequence. The gesture can be defined by either an input image or a short video clip. The task involves analyzing a test video and determining whether the desired gesture occurs. If the gesture is detected, the output frames will be annotated with the text "DETECTED" in bright green on the top right corner.

# Data Processing
The input data, including the desired gesture representation and the test video, are preprocessed and prepared for analysis using the following steps:

For the desired gesture representation:

**If provided as an image**: The image is resized to match the input size expected by the chosen model (224x224 pixels for EfficientNetB0). Preprocessing is applied to normalize the pixel values.

**If provided as a video**:For both desired gesture video and test video, Frames from the video are extracted and resized to the same dimensions as above. Preprocessing is applied to each frame individually.
For the gesture video: After converting each frame to a feature vector, the Mean is taken to convert it to a single feature vector and stored for comparison with test video later.

# Model Selection/Development
In this project, EfficientNetB0 is chosen as the model for gesture recognition due to its efficiency and good performance. EfficientNetB0 strikes a balance between model size and accuracy, making it suitable even for mobile applications with limited computational resources.

**For gesture image** : I am comparing the image feature vector to each frame of test video using cosine similarity. threshold is 0.8. I also tried using combination of mean and max but there was not any big difference.

**for gesture video ** : I have used the sliding window technique on the test video. the window is chosen to be of length of gestur video, frames are added to the window till window size become equal to gesture video size. then mean is taken. after that I am comparing feature vectors using the Distance time wrapping algorithm which is good for detecting gestures even if the speed of gestures in different.
I also tried using the cosine similarity function and it also gave satisfactory results.

# Detection Algorithm
The algorithm for gesture detection involves comparing the features extracted from the desired gesture representation with the features extracted from frames of the test video. The similarity between features is calculated using Dynamic Time Warping (DTW) for video gestures and cosine similarity for image gestures.

# Annotation
An annotation method is implemented to overlay text on the video frames where the gesture is detected. The text "DETECTED" is placed in bright green on the top right corner of the annotated frames.

# Documentation
In this project, I assumed that the desired gesture can be represented either by an image (.jpg, .png., .jpeg) or a short video clip. We utilized the EfficientNetB0 model for feature extraction due to its efficiency and good performance. For gesture detection, we employed DTW for video gestures and cosine similarity for image gestures.
**Challenges faced**:  Selecting an appropriate model and handling input data preprocessing efficiently. These challenges were addressed by carefully considering the trade-offs between model complexity and performance, as well as implementing preprocessing techniques to ensure consistency and compatibility with the chosen model.
Choosing between cosine similarity or DTW for video gesture detection was also a challenge. DTW takes a bit more time compared to cosine similarity calculation.
finding the right threshold took some trial and error. 

**Gesture Detection in Video Sequences**
This project aims to develop a prototype to detect a specific gesture within a video sequence. The gesture can be defined by either an input image or a short video clip. The task involves analyzing a test video and determining whether the desired gesture occurs. If the gesture is detected, the output frames will be annotated with the text "DETECTED" in bright green on the top right corner.

**Data Processing**
The input data, including the desired gesture representation and the test video, are preprocessed and prepared for analysis using the following steps:

For the desired gesture representation:

If provided as an image: The image is resized to match the input size expected by the chosen model (224x224 pixels for EfficientNetB0). Preprocessing is applied to normalize the pixel values.
If provided as a video: Frames from the video are extracted and resized to the same dimensions as above. Preprocessing is applied to each frame individually.
For the test video:
after converting each frame to a feature vector, Mean is taken to convert it to single vector.

Frames from the video are extracted and resized to match the input size expected by the chosen model. Preprocessing is applied to each frame individually just like gesture video.

# Gesture-detection

**Model Selection/Development**
In this project, EfficientNetB0 is chosen as the model for gesture recognition due to its efficiency and good performance. EfficientNetB0 strikes a balance between model size and accuracy, making it suitable even for mobile applications with limited computational resources.
For gesture image : i am comparing image feature vector to each frame of test video using cosine similarity. threshold is 0.8.
for gesture video : I have used sliding window technique on test video. window is chosen to be of length of gestur video, frames are added to window till window size become equal to gesture video size. then mean is taken. after that i am comparing feature vectors using Distance time wrapping algorithm which is good for detecting gestures even if speed of gestures in different.
I have also used cosine similarity function and it also gave satisfactory results.

**Detection Algorithm**
The algorithm for gesture detection involves comparing the features extracted from the desired gesture representation with the features extracted from frames of the test video. The similarity between features is calculated using Dynamic Time Warping (DTW) for video gestures and cosine similarity for image gestures.

**Annotation**
An annotation method is implemented to overlay text on the video frames where the gesture is detected. The text "DETECTED" is placed in bright green on the top right corner of the annotated frames.

**Documentation**
In this project, we assumed that the desired gesture can be represented either by an image (.jpg, .png., .jpeg) or a short video clip. We utilized the EfficientNetB0 model for feature extraction due to its efficiency and good performance. For gesture detection, we employed DTW for video gestures and cosine similarity for image gestures. 
**Challenges faced** :  Selecting an appropriate model and handling input data preprocessing efficiently. These challenges were addressed by carefully considering the trade-offs between model complexity and performance, as well as implementing preprocessing techniques to ensure consistency and compatibility with the chosen model.
Choosing between cosine similarity or DTW for video gesture detection was also a challange. DTW takes a bit more time compared to cosine similarity calculation.
finding the right threshold took some trial and error. 

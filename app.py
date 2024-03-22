import cv2
import numpy as np
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import os

model = MobileNetV2(weights='imagenet', include_top=False)
# def overlay_text(frame, text):
#     cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
def overlay_text(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    margin = 10
    thickness = 2
    color = (0, 255, 0)


    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    x = frame.shape[1] - text_width - margin
    y = margin + text_height

    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

# distance time wrapping but it was not working as expected. no idea why
def dtw(s, t):
    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.linalg.norm(s[i-1] - t[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])

    return dtw_matrix[n, m]

# for video i am using cosine similarity to compare the gestures
def load_gesture_representation(gesture_video_path,test_video_path, threshold=0.8): 
    cap_gesture = cv2.VideoCapture(gesture_video_path)
    features_gestures = []
    count = 0
    while(cap_gesture.isOpened()):
        ret, frame = cap_gesture.read()
        if not ret or count == 20:
            break
        frame = cv2.resize(frame, (224, 224))
        x = preprocess_input(np.expand_dims(frame, axis=0))
        features_gesture = model.predict(x)
        features_gestures.append(features_gesture)
        # count += 1
    cap_gesture.release()
    desired_gesture = np.mean(features_gestures, axis=0)
    desired_gesture_max = np.max(features_gestures, axis=0)
    desired_gesture = np.mean(desired_gesture, axis=(1, 2))
    print(desired_gesture.shape)
    print(desired_gesture_max.shape)
    # print(features_gestures.shape)
    window_frames = []
    cap = cv2.VideoCapture(test_video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        x = preprocess_input(np.expand_dims(frame, axis=0))
        features_frame = model.predict(x)
        window_frames.append(features_frame)
        if len(window_frames) > len(features_gestures):
            window_frames.pop(0)
        if len(window_frames) == len(features_gestures):
            window_frames_temp = np.mean(window_frames, axis=(0))
            similarity = dtw(np.mean(window_frames_temp, axis=(1,2)), desired_gesture)
            print("similarity : ")
            print(similarity)
            if similarity < 7:
                overlay_text(frame, "DETECTED")
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def load_image_gesture_representation(gesture_image_path, test_video_path, threshold=0.8):
    gesture = cv2.imread(gesture_image_path)
    gesture = cv2.resize(gesture, (224, 224))
    gesture = preprocess_input(np.expand_dims(gesture, axis=0))
    desired_gesture = model.predict(gesture)
    test_video_path = 'test.mp4'  # Update with your test video filename
    cap = cv2.VideoCapture(test_video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        x = preprocess_input(np.expand_dims(frame, axis=0))
        features_frame = model.predict(x)
        similarity = np.dot(features_frame.flatten(), desired_gesture.flatten()) / (np.linalg.norm(features_frame) * np.linalg.norm(desired_gesture))
        if similarity > threshold:
            overlay_text(frame, "DETECTED")
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


desired_gesture_path = 'w_d8.mp4'  # Update with your desired gesture image filename
test_path = 'women_rope.mp4'  # Update with your test video filename
file_extension = os.path.splitext(desired_gesture_path)[1]
if file_extension in ['.jpg','.jpeg','.png']:
    load_image_gesture_representation(desired_gesture_path, test_path, threshold=0.8)
else:
    load_gesture_representation(desired_gesture_path ,test_path, threshold=0.8)

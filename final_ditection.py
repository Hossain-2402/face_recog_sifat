from PIL import Image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import cv2
from keras.models import load_model
import numpy as np
import mediapipe as mp

# Load the trained model
model = load_model('facefeatures_new_model_final.h5')

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Function for face alignment using MediaPipe
def align_face(image, detection):
    # Extract the bounding box from the detection
    bbox = detection.location_data.relative_bounding_box
    ih, iw, _ = image.shape
    x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
    
    # Crop the face from the image
    cropped_face = image[y:y+h, x:x+w]
    
    # Resize to the desired size (224x224 for InceptionV3 model)
    aligned_face = cv2.resize(cropped_face, (224, 224))
    
    return aligned_face

# Start video capture
video_capture = cv2.VideoCapture(0)
np.set_printoptions(suppress=True, precision=6)

# Initialize MediaPipe Face Detection
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect faces
        results = face_detection.process(rgb_frame)
        
        # Initialize the person count
        person_count = 0
        
        # If faces are detected
        if results.detections:
            person_count = len(results.detections)
            
            # For each detected face, perform recognition
            for detection in results.detections:
                # Draw bounding box
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                
                # Align the face
                aligned_face = align_face(frame, detection)
                
                # Convert aligned face to RGB for PIL
                im = Image.fromarray(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB))
                img_array = np.array(im)
                
                # Preprocess the image before feeding to the model
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                
                # Predict the class (person)
                pred = model.predict(img_array)
                print(pred)
                
                # List of names including the Unauthorized class
                names = ['Hossain', 'Sifat', 'Unauthorized']
                
                # Find the predicted class with the highest confidence
                index = np.argmax(pred[0])
                confidence = np.max(pred[0])
                
                # Set a threshold for authorized persons
                threshold = 0.8  # Adjust based on your requirements
                if confidence > threshold:
                    name = names[index]
                else:
                    name = 'Unauthorized'
                
                # Display the result on the frame
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        else:
            # No faces detected
            cv2.putText(frame, "No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        
        # Display the number of persons found on the frame
        cv2.putText(frame, f"Number of persons found: {person_count}", (50, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
        
        # Show the video frame
        cv2.imshow('Video', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()

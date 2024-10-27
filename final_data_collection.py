
import os
import cv2
import numpy as np

os.chdir('C:/Users/zulka/Documents/Machine_Learning/face_recog(refreshed)')

faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0: 
        return None
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cropped_face = img[y:y+h, x:x+w]
        return cropped_face  # Return the cropped face (exiting after finding the first one)

    return None

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()

    extracted_face = face_extractor(frame)  # Updated to avoid calling face_extractor multiple times

    if extracted_face is not None:
        count += 1
        
        # Resize the extracted face to a fixed size
        face = cv2.resize(extracted_face, (400, 400))
        
        # Save the extracted face to the specified directory
        file_name_path = "./training_data/Hasan_dark/" + str(count) + "dark" + ".jpg"
        cv2.imwrite(file_name_path, face)
        
        # Display the face count on the image
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        
        # Display the cropped face
        cv2.imshow("Face Cropper", face)  

    else:
        print("Face Not found")
        pass

    # Exit loop when 'Enter' (cv2.waitKey(1) == 13 refers to 'Enter' button) is pressed or after 100 faces are saved
    if cv2.waitKey(1) == 13 or count == 100:
        break

cap.release()
cv2.destroyAllWindows()

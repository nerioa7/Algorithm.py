import face_recognition as fr
import numpy as np
from glob import glob
import cv2
import argparse

parser = argparse.ArgumentParser(description='Face Recognition')
parser.add_argument('--webcam', default=False, action='store_true')
args = parser.parse_args()
# Training 
print("[INFO]: Training")
encodings = list()
names = list()
files = glob("./Training/*")
print("[INFO]: Total Images : ", len(files))
for file in files:
    image = cv2.imread(file)
    faces = fr.face_locations(image)
    if not len(faces)>1:
        encoding = fr.face_encodings(image, known_face_locations=faces)[0]
        encodings.append(encoding)
        names.append(file.split("/")[-1].split(".")[0])
        print("[INFO]: ", file, " - Trained")
        

print("[INFO]: Total Trained : ", len(encodings))

if not args.webcam:
    test_files = glob("./Testing/*")
    print("[INFO]: Total Images for Testing : ", len(test_files))
    for file in test_files:
        image = cv2.imread(file)
        faces = fr.face_locations(image)
        encode = fr.face_encodings(image, known_face_locations=faces)
        for (top, right, bottom, left), face_encoding in zip(faces, encode):
            matches = fr.compare_faces(encodings, face_encoding)
            name = "Unknown"
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = fr.face_distance(encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = names[best_match_index]
            
            # Draw a box around the face
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        file_name = "Result/" + file.split("/")[-1]
        cv2.imwrite(file_name, image)
    
    exit()


cap = cv2.VideoCapture(0)
writer = None
(W, H) = (None, None)
frame_width = 640
frame_height = 360
out = cv2.VideoWriter('./Result/outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

process_this_frame = True
while True:
    if not process_this_frame:
        process_this_frame = not process_this_frame
        continue

    ret, frame = cap.read()
    process_this_frame = not process_this_frame
    frame = cv2.resize(frame, (640, 360))
    faces = fr.face_locations(frame)
    encode = fr.face_encodings(frame, known_face_locations=faces)

    for (top, right, bottom, left), face_encoding in zip(faces, encode):
        # See if the face is a match for the known face(s)
        matches = fr.compare_faces(encodings, face_encoding, tolerance=0.55)

        name = "Unknown"
        # Or instead, use the known face with the smallest distance to the new face
        face_distances = fr.face_distance(encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)
    out.write(frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
cap.release()
out.release()
cv2.destroyAllWindows()


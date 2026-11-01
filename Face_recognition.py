import cv2
import face_recognition
import numpy as np
import csv
import os
from datetime import datetime

video_cap = cv2.VideoCapture(0)

rudra_img = face_recognition.load_image_file("C:/Users/shova/Downloads/Face recognition photos/Rudra.jpg")
rudra_encoding = face_recognition.face_encodings(rudra_img)[0]

soumalya_img = face_recognition.load_image_file("C:/Users/shova/Downloads/Face recognition photos/Soumalya.jpg")
soumalya_encoding = face_recognition.face_encodings(soumalya_img)[0]

mihir_img = face_recognition.load_image_file("C:/Users/shova/Downloads/Face recognition photos/Mihir.jpg")
mihir_encoding = face_recognition.face_encodings(mihir_img)[0]

shovan_img = face_recognition.load_image_file("C:/Users/shova/Downloads/Face recognition photos/Shovan.jpg")
shovan_encoding = face_recognition.face_encodings(shovan_img)[0]

praneet_img = face_recognition.load_image_file("C:/Users/shova/Downloads/Face recognition photos/Praneet.jpg")
praneet_encoding = face_recognition.face_encodings(praneet_img)[0]

snehal_img = face_recognition.load_image_file("C:/Users/shova/Downloads/Face recognition photos/Snehal.jpg")
snehal_encoding = face_recognition.face_encodings(snehal_img)[0]

mohip_img = face_recognition.load_image_file("C:/Users/shova/Downloads/Face recognition photos/Mohip.jpg")
mohip_encoding = face_recognition.face_encodings(mohip_img)[0]

sundar_img = face_recognition.load_image_file("C:/Users/shova/Downloads/Face recognition photos/Sundar.jpg")
sundar_encoding = face_recognition.face_encodings(sundar_img)[0]

aritra_img = face_recognition.load_image_file("C:/Users/shova/Downloads/Face recognition photos/Aritra.jpg")
aritra_encoding = face_recognition.face_encodings(aritra_img)[0]

known_face_encoding = [
    rudra_encoding,
    soumalya_encoding,
    mihir_encoding,
    shovan_encoding,
    praneet_encoding,
    snehal_encoding,
    mohip_encoding,
    sundar_encoding,
    aritra_encoding
]

known_face_names = [
    "Rudrapratap De",
    "Soumalya Saha",
    "Mihir Banerjee",
    "Shovan Hazra",
    "Praneet Ghatak",
    "Snehal Mitra",
    "Mohip Ghosh",
    "Sundaram Nath",
    "Aritra Kar"
]

student = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s=True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    ret, video_data = video_cap.read()
    small_frame = cv2.resize(video_data,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            if name in known_face_names:
                if name in student:
                    student.remove(name)
                    print(student)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])    

    cv2.imshow("Recognise", video_data)

    if cv2.waitKey(1) & 0xFF == ord("x"):
        break
video_cap.release()    
cv2.destroyAllWindows()
f.close()

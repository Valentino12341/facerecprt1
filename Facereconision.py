import cv2,sys,numpy,os

(width, height)= (500,500)

face_cascade = cv2.CascadeClassifier("facerecfile.xml")
webcam = cv2.VideoCapture(0)
count= 1
while count <= 30:
    (_,image) = webcam.read()
    grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey,1.3,4)
    count +=1
print(faces)












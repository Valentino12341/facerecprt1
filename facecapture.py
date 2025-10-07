import cv2,sys,numpy,os

(width, height)= (500,500)

face_cascade = cv2.CascadeClassifier("facerecfile.xml")
webcam = cv2.VideoCapture(0)
count= 1
path = ("faces/lacasadepapel")##put in the name of the person



while count <= 30:
    (_,image) = webcam.read()
    grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey,1.3,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,h+y),(200,100,200),6)
        face = grey[y:y+h,x:x+w]
        cropface = cv2.resize(face,(500,500))
        cv2.imwrite(f"{path}/{count}.png",cropface)
    count +=1
    print(faces)


print(faces)



















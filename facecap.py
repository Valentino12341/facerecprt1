import cv2,sys,os
import numpy as np

face_cascade = cv2.CascadeClassifier("facerecfile.xml")
pathf = "faces"
(images,labels,names,id) = ([],[],{},0)

for (subdirs,dirs,files) in os.walk(pathf):
    for subdirs in dirs:
        names[id] = subdirs
        fullpath = os.path.join(pathf,subdirs)
        for filename in os.listdir(fullpath):
            print(filename)
            npath = fullpath+"/"+filename
            label = id
            images.append(cv2.imread(npath,0))
            labels.append(int(label))
        id += 1

(width,height) = (130,100)
images = np.array(images)
labels = np.array(labels)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images,labels)###
webcam = cv2.VideoCapture(0)
while True:
    (_,image) = webcam.read()
    thegrey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(thegrey,1.3,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,h+y),(40,30,20),7)
        face = thegrey[y:y+h,x:x+w]
        thecropface = cv2.resize(face,(130,100))
        prediction = model.predict(thecropface)
        cv2.rectangle(image,(x,y),(x+w,y+h),(40,50,60),8)
        if prediction[1]<100:
            cv2.putText(image,names[prediction[0]],(x+5,y-10),cv2.FONT_HERSHEY_PLAIN,2,(170,180,190),3)

        else:
            print("thats no u :(")

    cv2.imshow("TITLE",image)

    key = cv2.waitKey(10)
    if key ==27:
        break
#text = cv2.putText(char,"THE text",(30,50),font,1,color1,2,cv2.LINE_AA)
#cv2.imshow("g",text)



















import cv2,os
import csv
import numpy as np
from PIL import Image
import pandas as pd

recognizer = cv2.face.LBPHFaceRecognizer_create()  
faceCascade = cv2.CascadeClassifier('haarcascade.xml')

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids


def Take(path):
    id=int(input("enter id"))
    NAME=input("enter the name")
    faceCascade = cv2.CascadeClassifier('haarcascade.xml')
    video_capture = cv2.VideoCapture(path)
    
    i=0
    while (video_capture.isOpened()):
        try:
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                crop_img = frame[y:y+h, x:x+w]
                cv2.imwrite("images\ " + str(NAME) +"." + str(id) + '.' + str(i) + ".jpg",
                                    gray[y:y + h, x:x + w])
                i=i+1
            cv2.imshow('Video', frame)

        except:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            harcascadePath = "haarcascade.xml"
            detector = cv2.CascadeClassifier(harcascadePath)
            faces, ID = getImagesAndLabels("images")
            recognizer.train(faces, np.array(ID))
            recognizer.save("data.yml")
            print("trained")    
            with open('employee.csv', 'a+') as csvFile:
                writer = csv.writer(csvFile)
                row = [id, id, NAME]
                writer.writerow(row)
                print("saved")
                csvFile.close()
            break
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
            
    


    video_capture.release()
    cv2.destroyAllWindows()



def TrackImages(path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()  
    faceCascade = cv2.CascadeClassifier('haarcascade.xml')
    recognizer.read("data.yml")
    cam = cv2.VideoCapture(path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name']
    df = pd.read_csv("employee.csv")
    uu=[]
    unknown=[]
    unknowncount=0
    while (cam.isOpened()):
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
            print(conf)
            if (conf < 50):
                uu.append(conf) 
            else:
                print("trying to find user")
        c=(len(uu))
        if c==3:
            print("user verified")
            z=(df.loc[df['SERIAL'] == serial]['NAME'].values)
            print(str(z)+("::is authenticated"))
            break
        else:
            print("taking samples")
            unknowncount=unknowncount+1
            unknown.append(unknowncount)
            ch=len(unknown)
            print(ch)
            if len(unknown)==10:
                cv2.destroyAllWindows()
                print("user not found please register")
                break



     
        cv2.imshow('Taking Attendance', im)
        if (cv2.waitKey(1) == ord('q')):
            cv2.destroyAllWindows()
            Take()

    cam.release()
    cv2.destroyAllWindows()




      
      


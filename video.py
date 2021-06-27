import cv2
faceCascade = cv2.CascadeClassifier('haarcascade.xml')
video_capture = cv2.VideoCapture("test/97907.mp4")
i=0
while(video_capture.isOpened()):
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("hfcfh")
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
        cv2.imwrite("images\ " + "anand." + str(1) + '.' + str(i) + ".jpg",
                            gray[y:y + h, x:x + w])
        i=i+1
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

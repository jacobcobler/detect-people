import cv2

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
captureDevice = cv2.VideoCapture(0)

while True:
    rec, image = captureDevice.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    seq = classifier.detectMultiScale(gray, 1.05,6)

    for(x,y,w,h) in seq:
        cv2.rectangle(image,(x,y),(x+w, y+h), (255,0,0), 5)
        print("found")

    cv2.imshow('img', image)
    delay = cv2.waitKey(10) & 0xff
    if delay == 10:
        break

captureDevice.release()
cv2.destroyAllWindows()
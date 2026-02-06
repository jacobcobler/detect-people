import cv2
import time

classifier = cv2.CascadeClassifier("haarcascade_fullbody.xml")
if (classifier.empty):
        raise IOError("Error loading haarcascade_fullbody.xml")

captureDevice = cv2.VideoCapture(0)

t1 = time.time()
while True:
    rec, image = captureDevice.read()
    if not rec:
         break
    
    #Resize image
    image = cv2.resize(image, (640, 480))
    #Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    seq = classifier.detectMultiScale(gray, 1.05,6)

    for(x,y,w,h) in seq:
        cv2.rectangle(image,(x,y),(x+w, y+h), (255,0,0), 5)
        # GPIO.output(18,GPIO.HIGH)
        t2 = time.time()
        if (t2 - t1) > 1:
            print("found")
            t1 = time.time()

    cv2.imshow('img', image)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

captureDevice.release()
cv2.destroyAllWindows()
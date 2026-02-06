import cv2
import time

classifier = cv2.CascadeClassifier("haarcascade_fullbody.xml")
if classifier.empty():
        raise IOError("Error loading haarcascade_fullbody.xml")

captureDevice = cv2.VideoCapture(0)

COOLDOWN = 1.5
previous_time = 0

while True:
    rec, image = captureDevice.read()
    if not rec:
         break
    
    # Resize image
    image = cv2.resize(image, (640, 480))
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Adjust Contrast
    gray = cv2.equalizeHist(gray)

    persons = classifier.detectMultiScale(
         gray,
         scaleFactor=1.05,
           minNeighbors=6,
             minSize=(60, 120)) # decrease the chance of false positives
    
    persons_detected = False

    for(x,y,w,h) in persons:
        # Filter out thin boxes
        if w * h < 8000:
             continue
        
        persons_detected = True
        cv2.rectangle(image,(x,y),(x + w, y + h), (0, 255, 0), 3)

    current_time = time.time()
    if persons_detected and (current_time - previous_time > COOLDOWN):
         print("Person detecteed - Light On")
         last_time = current_time


    status = "People Detected" if persons_detected else "No People"
    color = (0, 255, 0) if persons_detected else (0, 0, 255)
    cv2.putText(image, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Pedestrian Detection", image)
    
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

captureDevice.release()
cv2.destroyAllWindows()
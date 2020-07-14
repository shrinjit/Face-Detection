import cv2,sys
imagePath = sys.argv[1]  # getting the image
cascPath = sys.argv[2]   # gettting the xml file 

faceCascade = cv2.CascadeClassifier(cascPath) # created a HARR cascade mandatory



image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     # reading the image 


# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

## the above method detectMultiScale is used to detect the images 


print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)



cv2.imshow("Faces found", image)
cv2.waitKey(0)
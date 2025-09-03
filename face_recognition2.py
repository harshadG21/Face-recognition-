import cv2,numpy,os

size=4
haar_file=haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
datasets=r'C:\Users\Admin\OneDrive\Desktop\projects\Face detection\datasets'
print('Training...')

(images,labels,names,id)=([],[],{},0)
(width,height)=(130,100)

for (root,dir,files) in os.walk(datasets):
    for subdir in dir:
        print(f" Found folder: {subdir}") 
        names[id] = subdir
        subjectpath= os.path.join(root,subdir)

        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            img=cv2.imread(path,0)

            if img is None:
                print(f"skipping unreadable image:{path}")
                continue

            img = cv2.equalizeHist(img)

            img=cv2.resize(img, (width, height))
            images.append(img)
            labels.append(id)

        print(f"Loaded {len([f for f in os.listdir(subjectpath)])} files for {subdir}")
        id +=1

if len(images)==0:
    print("No images found in datasets.check folder structure")
    exit()

images = numpy.array(images)
labels = numpy.array(labels)

print(f"Traing{len(images)} image across{len(names)} classes")
model=cv2.face.LBPHFaceRecognizer_create()
model.train(images,labels)
print("training Complete")

face_cascade=cv2.CascadeClassifier(haar_file)
cam=cv2.VideoCapture(0)
count=0

while True:
    (ret,im)=cam.read()
    if not ret:
        print("failed to grab frame")
        break

    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
        face = gray[y:y+h,x:x+w]
        face_resize=cv2.resize(face,(width,height))

        prediction=model.predict(face_resize)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
        if prediction[1]<70:
            cv2.putText(im,'%s - %.0f'%(names[prediction[0]],prediction[1]),
                        (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(names[prediction[0]])
            count=0
        else:
            count +=1
            cv2.putText(im,'Unknown',
                        (x-10,y-10),
                        cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
            if (count>100):
                print("Unknown person")
                cv2.imwrite("input.jpg",im)
                count=0

        cv2.imshow('OpenCV',im)
        key = cv2.waitKey(10)
        if key == 27:
            break

cam.release()
cv2.destroyAllWindows()


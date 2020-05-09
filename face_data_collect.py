# Write a Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcascades)
# Stores the Face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box (haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data
import cv2
import numpy as np

# initialize camera

cap=cv2.VideoCapture(0)

# face detection

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

#creating object of cascadeclassifier of face classifier type

# harcascade classifier is a classifier which is trained on a lot of facial data 

face_data=[] #to store data of face


dataset_path='./data/' 
#it create folder namely dataset_path directory
file_name=input("enter the name of person:")

# it is an empty folder on my project folder to store data of different person images
skip=0
while True:

	ret,frame=cap.read()

	if ret==False:

		continue
	gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	#faces=face_cascade.detectMultiScale(frame,1.3,5)

	faces=face_cascade.detectMultiScale(frame,1.3,5)

	# faces is of list of tuple where each tuple contain (x,y,w,h) x,y are cordinate of that point and w and h are width and height of image
	if len(faces)==0:
		continue
	faces=sorted(faces,key=lambda f:f[2]*f[3])

	# sort the list of faces by area area is (f[2]*f[3])
	
	#pick the last face (because it is the largest face acc to area)

	
	for face in faces[-1:]:

		x,y,w,h=face

		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		offset=10
		face_section=frame[y-offset:y+offset+h,x-offset:x+offset+w]
		face_section=cv2.resize(face_section,(100,100))

		skip+=1
		if skip%10==0:
			face_data.append(face_section)
			print(len(face_data))

		
		# extract (crop out the requird face) :region of interest
	cv2.imshow("Frame",frame)
	cv2.imshow("face section",face_section)
	key_pressed=cv2.waitKey(1) & 0xFF

	if key_pressed==ord('q'):

		break

#convert our face list array into a numpy array
face_data=np.array(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

# save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("data successfully sav1e at "+dataset_path+file_name+'.npy')
cap.release()
cv2.destroyAllWindows()

from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np
import alarm2 as alarm
from threading import Thread
from predict import pred

from apscheduler.schedulers.blocking import BlockingScheduler
import csv




tdata={}
sec=10
ta=[]
tb=[]
tcount=0
tyawn=0
tear=0
tearate=0
maxl=0
maxr=0
# with open('data.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Yawn", "Eye Blink", "Left Eye","Right Eye"])

def average(lst): 
    l=len(lst)
    if(l==0):
        return 0
    else:
        return sum(lst) /l

def cinc():
    # print("scheduler")
    global tcount,tyawn,tear,tearate,ta,tb
    tdata[tcount]=[tyawn,tear,average(ta),average(tb)]
    #print(pred(tdata[tcount]))
    x=pred(tdata[tcount])
    if(x[0]=="Drowsy"):
            alarm.playalarm()

    

    #print(len(ta),len(tb))
    # with open('data2.csv', 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     # writer.writerow(["Yawn", "Eye Blink", "Left Eye","Right Eye"])
    #     writer.writerow(tdata[tcount])
    tyawn=0
    tear=0
    ta=[]
    tb=[]
    tearate=0
    tcount+=1
    # print(tdata)
    return

def callsc():
    # print("callsc")
    scheduler = BlockingScheduler()
    scheduler.add_job(cinc, 'interval',args=[] ,seconds=10)
    scheduler.start()
    return


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
#cascade_path='haarcascade_frontalface_default.xml'
#cascade = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector()


def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts =             np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])

def mouth_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        return image, 0
    
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance

    #cv2.imshow('Result', image_with_landmarks)
    #cv2.imwrite('image_with_landmarks.jpg',image_with_landmarks)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


yawns = 0
yawn_status = False 


def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def eye_length(eye,d):
	global ta,tb
	# print("eye length")
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C=(A+B)/2
	if(d=="L"):
		ta+=[C]
	elif(d=="R"):
		tb+=[C]
	return

thresh = 0.23
frame_check = 5
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]


t1=Thread(target=callsc)
t1.start()
# cap=cv2.VideoCapture("fast.mp4")
# if (cap.isOpened()== False): 
#   print("Error opening video stream or file")
cap=cv2.VideoCapture(0)
flag=0
while(cap.isOpened()):
	# print("reading")
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	image_landmarks, lip_distance = mouth_open(frame)
	prev_yawn_status = yawn_status  
	if lip_distance > 30:
		yawn_status = True 
        
		cv2.putText(frame, "Subject is Yawning", (50,450), 
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
        

		output_text = " Yawn Count: " + str(yawns + 1)

		cv2.putText(frame, output_text, (50,50),
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
        
	else:
		yawn_status = False 
         
	if prev_yawn_status == True and yawn_status == False:
		yawns += 1
		tyawn+=1

	cv2.imshow('Live Landmarks', image_landmarks )
	cv2.imshow('Yawn Detection', frame )


    

    
	for subject in subjects:

		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)#converting to NumPy Array
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		eye_length(leftEye,"L")
		eye_length(rightEye,"R")
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		if ear < thresh:
			flag += 1
			tear+=1
			print (flag)
			if flag >= frame_check:
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#print ("Drowsy")
				# alarm.play.on=1
				# alarm.call()
		else:
			flag = 0
			# alarm.play.on=0
			# if(not alarm.play.on):
			# 	#print("\n\nDismissed\n\n")
			# 	pass
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
cap.release()
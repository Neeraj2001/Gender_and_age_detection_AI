import cv2
import math
import argparse

def CaptureFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (400, 400), [104, 117, 123], True, False)
    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for k in range(detections.shape[2]):
        confidence=detections[0,0,k,2]
        if confidence>conf_threshold:
            a=int(detections[0,0,k,3]*frameWidth)
            b=int(detections[0,0,k,4]*frameHeight)
            a1=int(detections[0,0,k,5]*frameWidth)
            b1=int(detections[0,0,k,6]*frameHeight)
            faceBoxes.append([a,b,a1,b1])
            
            cv2.rectangle(frameOpencvDnn, (a,b), (a1,b1), (10,100,100), int(round(frameHeight/200)), 10)
    return frameOpencvDnn,faceBoxes


parser=argparse.ArgumentParser()
parser.add_argument('--img')

args=parser.parse_args()

faceProto="ocv.pbtxt"
faceModel="ocv.pb"
genderProto="gender.prototxt"
genderModel="gender.caffemodel"
ageProto="age.prototxt"
ageModel="age.caffemodel"


MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-4)','(4-8)', '(8-12)', '(12-20)', '(20-32)', '(32-43)', '(43-53)', '(53-100)' ]
genderList=['HE IS MALE','SHE IS FEMALE']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

video=cv2.VideoCapture(args.img if args.img else 0)
padding=20
while cv2.waitKey(1)<0:
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg,faceBoxes=CaptureFace(faceNet,frame) # detects number of faces in the frame
    if not faceBoxes:
        print("Face is' not detected")

    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')
        print("\n");
        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(resultImg, f'{gender} {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting the age and gender", resultImg)

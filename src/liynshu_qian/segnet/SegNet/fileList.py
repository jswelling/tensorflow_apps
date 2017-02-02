import os

LOAD_DIR="/pylon1/sy4s8lp/anniez/project/caffe-segnet/tutorial/CamVid/train.txt"

prefix="/pylon1/sy4s8lp/anniez"

filenames=open(LOAD_DIR,"r")
nameL=[]
for line in filenames:
	img,label=line.split(" ")
	nameL.append(prefix+img[19:])
	nameL.append(prefix+label[19:-1])
	#print (img,"YAYAYAYAYAY",label,"wowowowowowowow")
print nameL

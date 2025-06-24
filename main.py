# import Threading
import threading
import FaceDetection
import voice
import time 
import sys
import collections
# import FaceShow
from tkinter import *
from PIL import Image, ImageTk 
import cv2
import face_recognition

   
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
(knowFaceName,knowFaceEncoder) = FaceDetection.EncodeFaceData()
counter = 0


def FaceRecognition(label): 
    
    global knowFaceEncoder, knowFaceName, face_cascade, counter

    
    ## initialize the haarcascade for face detection 
    
    ## take input from laptop camera
    cap = cv2.VideoCapture(0)

    ## Create a window 
    # window = "Video Detection"
    # cv2.namedWindow(window)


    ## created loop for continuous recognition 
    while 1 :
        # print(cap.isOpened())

        hasFrame, frame = cap.read()
        frame = cv2.flip(frame,1)
        frame2 = frame.copy()
        
        ## check we have frame or not
        if hasFrame :
            
            ## Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            ## detect faces from frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=11)
            
            # cv2.imshow(window,frame)
            # cv2.imshow("newImage",frame2)
            
            ## check we have detect the face 
            if len(faces) >= 1:
                
                ## taking the first face form faces
                faces = faces[0]
                # print(faces)
                
                # taking the rectangle points 
                x,y,w,h = faces
                
                # print("dimentions are ",x,y,w,h)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # taking the face part from the frame  
                img = frame2[x-100 : x+w+100 , y - 100 : y+h+100 , :]
            
                try :
                    ## converting the BGR img to RGB Image
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                except:
                    pass
                
                # Encode the face data 
                encodeImg = face_recognition.face_encodings(img)
                # print("Encoded image",encodeImg)
                # cv2.imshow("Cropped face", img)
                
                
                try :
                    encodeImg = encodeImg[0]
                    # print("this is encoded image",encodeImg)
                except Exception as e:
                    pass
                
                matchs = []
                
                ## Compare encoded image to known encoded image 
                for knowImg in knowFaceEncoder:
                    # print(knowImg)
                    # print("Start to compares faces")
                    try :
                        match = face_recognition.compare_faces([knowImg],encodeImg,tolerance=0.4)
                        matchs.extend(match)
                        
                    except Exception as e :
                        # print("Error 1")
                        pass
                        
                # print("the finded matchis are ",matchs)
                
                ## Check if we get match or not.
                
                try :
                    # print("collecting the true match")
                    index = matchs.index(True)
                    # print("the index of the image ")
                    nameOfPerson = knowFaceName[index]
                    
                    # if getName() == nameOfPerson :
                    #     pass
                    # else :
                    FaceDetection.ObjectName.setName =  nameOfPerson
                    
                    # print(FaceDetection.ObjectName.getName, " with out exception")
                    
                except Exception as e:
                    # if getName() == "Human":
                    #     pass
                    # else : 
                    nameOfPerson = "Human"
                    FaceDetection.ObjectName.setName = nameOfPerson
                    
                    # print(FaceDetection.ObjectName.getName, " with exception " )
                    
                
                ## put text on the frame
                cv2.putText(frame,nameOfPerson,(x-10, y-10),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=1,color=(0,255,0),thickness=2)
                # print("face recognize as ",name)
            
            else :
                
                counter = counter + 1
                
                if counter > 10 :
                    FaceDetection.ObjectName.setName ="PersonGone"
                    # print(getName(), " with exception " )
                    counter = counter + 1
                
  
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

            # Capture the latest frame and transform to image 
        captured_image = Image.fromarray(opencv_image) 

            # Convert captured image to photoimage 
        photo_image = ImageTk.PhotoImage(image=captured_image) 

            # Displaying photoimage in the label 
        # face_frame.photo_image = photo_image 

            # Configure image in the label 
        label.configure(image=photo_image) 
        
        label.image = photo_image

        # # Repeat the same process after every 10 seconds 
            
        
    # cap.release()    
    # cv2.destroyAllWindows()

# def show():
    # FaceRecognition()
    
# show()
flag = False

def nameAndCommnunication(label):
    global knowFaceName, flag
    
    print("Know face is : ",knowFaceName)
    counter = 0
    faces = []
    while 1 :
        
        if flag: 
            label.config(text = 'Waiting for you')

            time.sleep(1)
            faceName = FaceDetection.getName()
            
            print("face name is ",faceName) 
            if faceName == "NoPerson" or faceName =="PersonGone":
                continue
            else :
                faces.append(faceName)
                counter += 1
                       
            if (faceName in knowFaceName or faceName == "Human") and counter > 5 :
                print(faces)
                faceName = collections.Counter(faces).most_common()[0][0]
                message = "communication start with " +faceName
                label.config(text = message)
                voiceThread = threading.Thread(target=voice.voiceMain,args=(faceName,))
                voiceThread.start()
                voiceThread.join()
                message = "communication Ends with " +faceName
                label.config(text = message)
                
                # voice2.voiceMain(faceName)
                counter = 0 
                faces = []
                
            else :
                pass
            
        else :
            break
    
        
def main():
    global flag
    
    window = Tk()
    window.title("Intelligent Assistant With Face Recognition")


    width= window.winfo_screenwidth()               
    height= window.winfo_screenheight()               
    window.geometry("%dx%d" % (width, height))


    communication_label = Label(text="Communication")
    communication_label.grid(row=0, column=1)

    face_frame = Label(width=640, height=480)
    face_frame.grid(row=0, column=0)
    face_frame.grid_propagate(False)
    face_frame.grid_rowconfigure(0, weight=1)  
            
    window.bind('<Escape>', lambda e: window.quit())
    
    # print(1)
    
    faceProcess = threading.Thread(target=FaceRecognition,args=(face_frame,))

    faceCommunication = threading.Thread(target=nameAndCommnunication,args=(communication_label,))

    faceProcess.start()
    
    if not faceProcess.is_alive():
        flag = False
        sys.exit()
    else :
        flag = True


    faceCommunication.start()
    
    window.mainloop()

    faceProcess.join()
    sys.exit()
    
if __name__ =="__main__":
    
    main()
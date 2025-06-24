import listenVoice
import ModelPrediction
import say
import whisper
import FaceDetection

import speech_recognition as sr

#recognizer = sr.Recognizer()
#audio = recognizer.listen()

## Load the whisper base model for offline communication.
whisper.load_model("base")

## Created dict for remember user 
face_lifetime={}

## Core Voice communication function
def voiceMain(nameOfPerson):
    """
    this function is created for communication with user.
    it take input face name and 
    """

    faceName = nameOfPerson
    
    ## this counter is used for track if user is not speeck 5 time then it terminate
    counter  = 0
    counterForPerson = 0 
    great = True
    
    while(1):
        name = FaceDetection.getName()
        print("the name detected is ",name)
        print("the name we gat is ",faceName)
        
        if faceName != name :
            counterForPerson += 1
            pass
        
        else :
            counterForPerson = 0
            
        if counterForPerson >= 2 :
            say.SpeakText('Come back soon! ' + faceName)
            great = True
            break


        if great  and faceName != "Human":
            
            if faceName not in face_lifetime.keys():
                face_lifetime[faceName] = 0
                say.SpeakText('welcome ' + faceName)
                great = False 
                
            elif faceName in face_lifetime.keys():
                
                say.SpeakText('welcome back' + faceName)
                great = False 
                    
        elif  great  and faceName == "Human" :
            say.SpeakText('welcome Human')
            great = False 
        
        
        ans = ""
        
        # listen.is_connected()
        
        text = listenVoice.capture_voice_input()
        
        ## pass text to the model
        print("the text is : ",text)
        
        if text =="":
            counter = counter + 1
            
            if counter > 5 :
                say.SpeakText("i dont get anything now i am turning off.")
                break

            else :
                say.SpeakText(f"speack something {faceName}. otherwise i am going to stop")
                continue        
        
        if text.lower() in ["good morning","good night","good afternoon","good evening"]:
            if text.lower() != "good night":
                say.SpeakText(text + " "+ faceName)
                continue
            else :
                say.SpeakText(text + " "+ faceName)
                break
       
        
        ans = ModelPrediction.chatbot_response(text)
        counter = 0
        print("model output ",ans)
        
        say.SpeakText(ans)
        
        if ans in  ["Sad to see you go :(","Talk to you later","Goodbye!","Come back soon!"]:
            break
        #SpeakText("text")

    print("Done")

# while 1 :
#     voiceMain()
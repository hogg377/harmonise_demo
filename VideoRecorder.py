import cv2
class VideoRecorder():
    # (adapted from https://www.geeksforgeeks.org/create-a-screen-recorder-using-python/)
    def __init__(self):
        self.fps = 60
        self.resolution = (1920, 1080)
        self.codec =  cv2.VideoWriter_fourcc(*"XVID")
        self.filename = "Recording.avi"
        self.out = []
    #end function

    def setConfig(self, fname = "Recording.avi", fps=60, resolution=(1920, 1080), codec="XVID"):
        self.fps = fps
        self.resolution = resolution
        self.codec = cv2.VideoWriter_fourcc(*codec)
        self.filename = fname
    #end function


    def startRecorder(self):
        self.out = cv2.VideoWriter(self.filename, self.codec, self.fps, self.resolution)


    def grabScreen(self, frame):
        #frame is a numpy array
        # Convert it from BGR(Blue, Green, Red) to
        # RGB(Red, Green, Blue)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Write it to the output file
        self.out.write(frame)
    #end function


    def stopRecorder(self):
        # Release the Video writer
        self.out.release()
        # Destroy all windows
        cv2.destroyAllWindows()
    #end function


    def main(self):
        #TODO: Create a function to record the current screen and output a live preview
        # import numpy as np
        # import 

        # # Create an Empty window
        # cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
        
        # # Resize this window
        # cv2.resizeWindow("Live", 480, 270)

        # self.startRecorder()
        
        # while True:
        #     # Take screenshot using PyAutoGUI
        #     img = pyautogui.screenshot()
        
        #     # Convert the screenshot to a numpy array
        #     frame = np.array(img)
        
        #     # Convert it from BGR(Blue, Green, Red) to
        #     # RGB(Red, Green, Blue)
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #     # Write it to the output file
        #     out.write(frame)
            
        #     # Optional: Display the recording screen
        #     cv2.imshow('Live', frame)
            
        #     # Stop recording when we press 'q'
        #     if cv2.waitKey(1) == ord('q'):
        #         break
        
        # # Release the Video writer
        # self.out.release()
        
        # # Destroy all windows
        # cv2.destroyAllWindows()
        return 1




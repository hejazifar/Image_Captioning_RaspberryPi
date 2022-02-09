import cv2

class Camera():
    def __init__(self, width = 2592, height = 1944):
       self.width = width
       self.height = height
       self.cap = None
       self.ret = None
       self.frame = None
       self.path = None
       
    def capture(self, save = False):   
        # open camera
        self.cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)
        # set dimensions
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # take frame
        self.ret, self.frame = self.cap.read()
        # release camera.
        self.cap.release()
        if save and self.ret:
            # write frame to file
            self.path ='image1.jpg'
            cv2.imwrite(self.path, self.frame)
            return self.path        
        return self.frame

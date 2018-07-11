from darkflow.net.build import TFNet
import cv2 as cv
import argparse
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


class ObjDetector:
    def __init__(self, label, cfg='yolo.cfg', weights='yolo.weights', ):
        options = {'model': 'yolo.cfg', 'load': 'yolo.weights', 'threshold': 0.1, 'gpu': 1}
        self.label = label
        with HiddenPrints():
            self.tfnet = TFNet(options)
    

    def detect(self, image):
        img = cv.imread(image)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        with HiddenPrints():
            results = self.tfnet.return_predict(img)

        for obj in results:
            if obj['label']==self.label and obj['confidence']>0.3:
                x = (obj['topleft']['x']+obj['bottomright']['x'])/2
                y = (obj['topleft']['y']+obj['bottomright']['y'])/2
                return x, y
        return 0, 0


    def inROI(self, image, roi):
        coord = self.detect(image)
        if coord[0]:
            if (roi[0][0]<=coord[0]<=roi[1][0]) and (roi[1][1]<=coord[1]<=roi[0][1]):
                return True
        return False



def mix(vid1, roi1, vid2, roi2, out):
    cap1 = cv.VideoCapture(vid1)
    cap2 = cv.VideoCapture(vid2)
    objd = ObjDetector('sports ball')
    writer = None
    first = True

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not writer:
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            (h, w) = frame1.shape[:2]
            fps = cap1.get(cv.CAP_PROP_FPS)
            writer = cv.VideoWriter(out, fourcc, fps, (w, h), True)

        if ret1 and ret2:
            if first:
                in1 = objd.inROI(frame1, roi1):
                if in1:
                    writer.write(frame1)
                    cv.imshow('Frame', frame1)
                    continue
            else:
                in2 = objd.inROI(frame2, roi2):
                if in2:
                    writer.write(frame2)
                    cv.imshow('Frame', frame2)
                    first = True
                    continue
            
            if not (in1 or in2):
                if first:
                    writer.write(frame1)
                    cv.imshow('Frame', frame1)
                    continue
                else:
                    writer.write(frame2)
                    cv.imshow('Frame', frame2)
                    first = True
                    continue
            if in1:
                writer.write(frame1)
                cv.imshow('Frame', frame1)
                continue
            if in2:
                writer.write(frame2)
                cv.imshow('Frame', frame2)
                first = True
                continue


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-i', '--image', required=True, help='Image file', type=str)
#     args = parser.parse_args()
#     print(detect(args.image))


if __name__=='__main__':
    mix('2018-06-20_20h12__1.mp4', roi1, '2018-06-20_20h12__2.mp4', roi2, 'out.avi')

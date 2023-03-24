from tkinter import *
import cv2
from PIL import Image, ImageTk
import argparse
import numpy as np
import predict
import pyzbar.pyzbar as pyzbar

global_flag = 1


class yolo():
    def __init__(self, confThreshold, nmsThreshold):
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.inpWidth = 416
        self.inpHeight = 416
        self.net = cv2.dnn.readNet('qrcode-yolov3-tiny.cfg', 'qrcode-yolov3-tiny.weights')

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=4)

        label = '%.2f' % conf
        label = '%s:%s' % ('qrcode', label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return frame

    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        indices = indices.flatten().tolist()
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)

    def detect(self, srcimg):
        blob = cv2.dnn.blobFromImage(srcimg, 1/255.0, (self.inpWidth, self.inpHeight), [0, 0, 0], swapRB=True, crop=False)
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        self.postprocess(srcimg, outs)
        return srcimg


#-------------------------------------------------------------------分割符-------------------------------------------------------------------------
def take_snapshot():
    global global_flag
    if global_flag == 1:
        global_flag = global_flag+1
        print("正在切换为DeblurGANv2去模糊")
    else:
        global_flag = global_flag-1
        print("正在切换为yolo3检测")
    print("模式切换 顺序为：1.基于YOLOv3-Tiny的二维码检测 2.基于DeblurGANv2和Zbar的模糊二维码识别")


def video_loop():

    success, frame = camera.read()  # 从摄像头读取照片
    if success:
        cv2.imwrite("frame.jpg", frame)
        if global_flag == 1:
            try:
                srcimg = cv2.imread('frame.jpg')
                srcimg = yolonet.detect(srcimg)
                out = srcimg
            except:
                out = frame
        else:
            video = predict.main_RT(frame)
            gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义一个核
            shape = cv2.filter2D(gray, -1, kernel=kernel)
            imgEqu1G = cv2.equalizeHist(shape)
            ret, binary = cv2.threshold(imgEqu1G, 180, 255, cv2.THRESH_BINARY)
            barcodes = pyzbar.decode(binary)
            for barcode in barcodes:
                (x, y, w, h) = barcode.rect
                cv2.rectangle(video, (x, y), (x + w, y + h), (0, 255, 0), 2)
                barcodeData = barcode.data.decode("utf-8")
                barcodeType = barcode.type
                text = "{}".format(barcodeData)
                cv2.putText(video, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print("[扫描结果] 二维码类别： {0} 内容： {1}".format(barcodeType, barcodeData))
            out = video
        #out =frame
        #cv2.waitKey(100)
        cv2image = cv2.cvtColor(out, cv2.COLOR_BGR2RGBA)  # 转换颜色从BGR到RGBA
        current_image = Image.fromarray(cv2image)  # 将图像转换成Image对象
        imgtk = ImageTk.PhotoImage(image=current_image)
        panel.imgtk1 = imgtk
        panel.config(image=imgtk)
        root.after(1, video_loop)


camera = cv2.VideoCapture(0)  # 摄像头
parser = argparse.ArgumentParser()
parser.add_argument('--confThreshold', default=0.6, type=float, help='confThreshold')
parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nmsThreshold')
args = parser.parse_args()
yolonet = yolo(args.confThreshold, args.nmsThreshold)
root = Tk()
root.title("运动模糊二维码检测与识别")
# root.protocol('WM_DELETE_WINDOW', detector)


panel = Label(root)  # initialize image panel
panel.pack(padx=10, pady=10)
# root.config(cursor="arrow")
btn = Button(root, text="点击切换下一个模式", command=take_snapshot)
btn.pack(fill="both", expand=True, padx=10, pady=10)

video_loop()

root.mainloop()
# 当一切都完成后，关闭摄像头并释放所占资源
camera.release()
cv2.destroyAllWindows()
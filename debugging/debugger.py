from darkflow.net.build import TFNet
import cv2
import matplotlib.pyplot as plt
from debugging.boxDrawer import drawPredictedObjects, showConfidence, drawFirstPrediction

options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}

tfnet = TFNet(options)

imgcv = cv2.imread("sample_img/sample_dog.jpg")
result, boxes = tfnet.return_predict(imgcv)

imgcvCopy = imgcv

drawFirstPrediction(boxes, imgcvCopy)
drawPredictedObjects(result, imgcvCopy)
showConfidence(result, imgcvCopy)

cv2.imwrite('sample_img/out/sample_dog_computed.jpg', imgcvCopy)
imgplot = plt.imshow(imgcv)
plt.show()

print(result)
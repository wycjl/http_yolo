import web
import json
import numpy as np
import cv2
import io
from PIL import Image
from yolo import YOLO # yolo目标检测
import keras # 深度学习框架
urls = ('/', 'Upload')
keras.backend.clear_session()
FLAGS = {}
detector = YOLO(**(FLAGS))

class Upload:
    def GET(self):
        return """<html><head></head><body>
<form method="POST" enctype="multipart/form-data" action="">
<input type="file" name="file" />
<br/>
<input type="submit" />
</form>
<img src='localhost:8080/test.jpg' />
</body></html>"""
    #
    def POST(self):
        x = web.input(file={})
        print(x['file'].file.name)
        buffer = io.BytesIO(x['file'].file.read()).getbuffer()
        image = cv2.imdecode(np.frombuffer(buffer, np.uint8),cv2.IMREAD_COLOR) #解码
        image = np.array(image)
        cv2.imwrite('test.jpg',image) # 原图
        temp = Image.open('test.jpg')

        result = detector.detect_image(temp) # 进行识别

        result = np.asarray(result)
        result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
        cv2.imwrite('test1.jpg',result) # 识别后的结果用opencv保存，文件名为test1.jpg
        detector.close_session()
        # 显示图片有问题，不知道怎么显示本地图片，test1.jpg就是处理后的图片
        return """<html><head></head><body>
                <img src='localhost:8080/test1.jpg'>
                </body></html>"""


if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()
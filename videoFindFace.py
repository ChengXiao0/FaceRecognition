import cv2
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
# 打开视频捕获设备
video_capture = cv2.VideoCapture(0)
while True:
    # 读视频帧
    ret, frame = video_capture.read()
    # 转为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 调用分类器进行检测
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    # 画矩形框
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 显示视频
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 关闭摄像头设备
video_capture.release()
# 关闭所有窗口
cv2.destroyAllWindows()

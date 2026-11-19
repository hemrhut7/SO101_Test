import cv2

# 填入剛剛查到的兩支相機編號
cam_ids = [0, 2]
caps = {}

for cid in cam_ids:
    cap = cv2.VideoCapture(cid)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # 建議強制使用 MJPG 格式，避免多支相機同時開啟時 USB 頻寬不足導致黑屏
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    caps[cid] = cap

while True:
    for cid, cap in caps.items():
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera {cid}", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()

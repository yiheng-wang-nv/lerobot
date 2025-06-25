import cv2

cap = cv2.VideoCapture(0)  # 0 is usually the default camera

cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)  # Allow window resizing
cv2.resizeWindow('Camera Feed', 640, 480)          # Set initial window size

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Camera Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import os                                                       
import cv2
from ultralytics import YOLO

face_model = YOLO(os.environ.get("YOLO_WEIGHTS_PATH", "weights/yolov8n-face.pt"))   

def detect_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("이미지를 찾을 수 없습니다.")
        return None
    
    h, w = img.shape[:2]
    
    results = face_model(img, verbose=False)[0]
    
    if len(results.boxes) == 0:
        print("얼굴을 감지하지 못했습니다.")
        return None
    
    boxes = results.boxes.xyxy.cpu().numpy()
    areas = [(x2-x1)*(y2-y1) for x1,y1,x2,y2 in boxes]
    x1, y1, x2, y2 = map(int, boxes[areas.index(max(areas))])
    
    face_ratio = ((x2-x1) * (y2-y1)) / (w * h)
    face_crop = img[y1:y2, x1:x2]
    
    print(f"bbox: ({x1}, {y1}, {x2}, {y2})")
    print(f"face_ratio: {face_ratio:.4f}")
    
    return face_crop, face_ratio

if __name__ == "__main__":
    result = detect_face('/workspace/capstone_pipeline/test_images/test_sulyoon.jpg')
    if result:
        face_crop, face_ratio = result
        print("detect_face 완료!")
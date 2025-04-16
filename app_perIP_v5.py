from flask import Flask, request, jsonify, render_template, Response, send_file
import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
import threading
import json
from datetime import datetime
import io
import requests

app = Flask(__name__)

# YOLOv8 모델 로드
person_model = YOLO('best_person.pt').to('cpu')  # 사람 인식용 경량 모델 사용
cars_model = YOLO('best_cars.pt').to('cpu')  # 차량 인식용 경량 모델 사용
total_model = YOLO('yolov8n.pt').to('cpu')  # cpu로 모델 로드

p_ip = ['192.168.50.16'] # 사람 인식용 IP 리스트
v_ip = ['192.168.50.17'] # 차량 인식용 IP 리스트 

# 최신 이미지 및 결과 저장용 변수
latest_images = {}
latest_results = {}
image_lock = threading.Lock()
results_lock = threading.Lock()

# 위험 이벤트 저장
events = []
events_lock = threading.Lock()

# 카메라 IP 저장
camera_ip = None
camera_ips = {}  
camera_ips_lock = threading.Lock()

# 위험 감지 함수 수정
def detect_danger(image):
    global person_model
    global cars_model
    global total_model
    global latest_results
    global camera_ip
    global p_ip, v_ip
    
    # YOLOv8로 객체 감지
    if camera_ip in p_ip:
        model = person_model
    elif camera_ip in v_ip:
        model = cars_model
    else:
        model = total_model
    results = model(image)
    
    # 결과 해석
    persons = []
    vehicles = []
    
    # 인식된 객체 정보 저장
    all_objects = []
    
    for r in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = r
        cls = int(cls)
        
        if not cls in [0, 1, 2, 3, 5, 7] or conf < 0.3:  # 사람, 자전거, 자동차, 오토바이, 버스, 트럭만 감지
            continue
            
        label = model.names[cls]
        confidence = float(conf)
        
        all_objects.append({
            "type": label,
            "confidence": round(confidence, 2),
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        })
        
        # 클래스 0: 사람, 클래스 2,3,5,7: 차량 관련 클래스
        if camera_ip in p_ip and cls == 0 and conf > 0.5:
            persons.append([int(x1), int(y1), int(x2), int(y2)])
        elif camera_ip in v_ip and cls in [1, 2, 3, 5, 7] and conf > 0.5:  # bycycle, car, motorcycle, bus, truck
            vehicles.append([int(x1), int(y1), int(x2), int(y2)])
    
    # 위험 판단 로직: 사람이나 차량이 감지되면 위험으로 판단 (수정된 부분)
    danger = False
    danger_message = ""
    event_image_filename = None  
    
    if persons:
        danger = True
        danger_message = "사람 감지. LED 경고!"
    
    if vehicles:
        danger = True
        danger_message = "차량 감지. LED 경고!"
    
    if persons and vehicles:
        danger = True
        danger_message = "사람과 차량이 모두 감지되었습니다. 주의하세요!"
    
    # 인식 결과 저장
    with results_lock:
        latest_results = {
            "danger": danger,
            "message": danger_message,
            "persons": len(persons),
            "vehicles": len(vehicles),
            "objects": all_objects
        }
    
    # 인식 결과가 포함된 이미지 생성
    annotated_img = draw_detection_results(image.copy(), all_objects, danger)
    
    if danger:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        event_image_filename = f"event_{timestamp}.jpg"
        events_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "events")
        os.makedirs(events_dir, exist_ok=True)  # Ensure the events directory exists
        image_path = os.path.join(events_dir, event_image_filename)
        cv2.imwrite(image_path, annotated_img)  # Save the annotated image
        latest_results["image_url"] = f"/image/{event_image_filename}"  # Add image URL to results
    
    return latest_results, annotated_img

# 인식 결과를 이미지에 표시하는 함수
def draw_detection_results(image, objects, danger):
    global camera_ip
    global p_ip, v_ip

    for obj in objects:
        label = obj["type"]
        confidence = obj["confidence"]
        bbox = obj["bbox"]
        
        x1, y1, x2, y2 = bbox
        
        # 객체 유형에 따라 색상 설정
        if camera_ip in p_ip and label == "person":
            color = (0, 255, 0)  # 초록색
        elif camera_ip in v_ip and label in ["bycycle", "car", "motorcycle", "bus", "truck"]:
            color = (0, 0, 255)  # 빨간색
        else:
            color = (255, 255, 0)  # 노란색
            
        # 바운딩 박스 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 라벨 텍스트
        text = f"{label}"
        #text = f"{label} {confidence:.2f}"
        cv2.putText(image, text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

@app.route('/detect', methods=['POST'])
def detect():
    global camera_ip
    global p_ip, v_ip
    
    try:

        # URL 파라미터에서 IP 주소 가져오기
        camera_ip = request.args.get('ip')
        print(f"카메라 IP from URL: {camera_ip}")

        # 이미지 데이터는 request.data에서 직접 가져오기
        img_data = request.data

        if not img_data:
            return jsonify({"error": "이미지 데이터가 없습니다.", "danger": False}), 400

        # 카메라 IP 업데이트
        if camera_ip:
            with camera_ips_lock:
                camera_ips[camera_ip] = datetime.now()

        try:
            # 바이너리 이미지 데이터를 직접 디코딩
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"이미지 디코딩 오류: {str(e)}")
            return jsonify({"error": "이미지 처리 실패", "danger": False}), 400

        if img is None:
            return jsonify({"error": "이미지를 처리할 수 없습니다.", "danger": False}), 400

        # 이미지 상하 반전 (ESP32-CAM 카메라 방향 수정)
        img = cv2.flip(img, -1)
        
        # 이미지 크기 조정 (320x240으로 리사이즈)
        img = cv2.resize(img, (320, 240))

        # 카메라별 최신 이미지 업데이트
        with image_lock:
            latest_images[camera_ip] = img.copy()

        # 위험 감지
        result, annotated_img = detect_danger(img)

        # 위험한 경우 이벤트 기록
        if result["danger"]:
            try:
                if int(result["persons"]) > 0:
                    arduino_url = "http://" + v_ip[0]
                elif int(result["vehicles"]) > 0:
                    arduino_url = "http://" + p_ip[0]

                print(f"Alert to Arduino at: {arduino_url}")
                print(f"Detection from camera: {camera_ip}")

                params = {
                    'danger': 1 if result["danger"] else 0,
                    'person': result["persons"],
                    'vehicle': result["vehicles"],
                    'message': result["message"]
                }
                requests.get(arduino_url, params=params, timeout=0.1)
            except requests.exceptions.RequestException:
                print("Failed to send alert to Arduino")

            # 위험 이벤트 기록
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 이미지 저장
            path = os.path.dirname(os.path.abspath(__file__))
            filename = path + f"/events/event_{timestamp.replace(' ', '_').replace(':', '-')}.jpg"
            os.makedirs(path + "/events", exist_ok=True)
            cv2.imwrite(filename, annotated_img)

            with events_lock:
                events.append({
                    "timestamp": timestamp,
                    "message": result["message"],
                    "image": filename
                })

                # 최대 100개 이벤트만 저장
                if len(events) > 100:
                    events.pop(0)

        # 명확한 JSON 형식으로 응답
        response_data = {
            "danger": 0,
            "message": result["message"],
            "persons": result["persons"],
            "vehicles": result["vehicles"]
        }

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e), "danger": False}), 400

    return jsonify(response_data)

@app.route('/events', methods=['GET'])
def get_events():
    with events_lock:
        for event in events:
            if "image" in event:
                event["image_url"] = f"/events/{os.path.basename(event['image'])}"
        return jsonify(events)

# 카메라별 비디오 스트림 생성
def generate_frames_for_camera(camera_ip):
    global p_ip, v_ip
    while True:
        with image_lock:
            if camera_ip not in latest_images:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"Camera {camera_ip} waiting...", 
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                frame = latest_images[camera_ip].copy()
                
                # 실시간 바운딩 박스와 라벨 표시
                with results_lock:
                    if latest_results and "objects" in latest_results:
                        for obj in latest_results["objects"]:
                            label = obj["type"]
                            confidence = obj["confidence"]
                            bbox = obj["bbox"]
                            x1, y1, x2, y2 = bbox
                            
                            # 객체 유형에 따라 색상 설정
                            if camera_ip in p_ip and label == "person":
                                color = (0, 255, 0)  # 초록색 (사람)
                            elif camera_ip in v_ip and label in ["bicycle", "car", "motorcycle", "bus", "truck"]:
                                color = (0, 0, 255)  # 빨간색 (차량)
                            else:
                                color = (255, 255, 0)  # 노란색 (기타)
                            
                            if (camera_ip in p_ip and label == "person") or (camera_ip in v_ip and label in ["bicycle", "car", "motorcycle", "bus", "truck"]):                            
                                # 바운딩 박스와 라벨 그리기
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                text = f"{label} {confidence:.2f}"
                                cv2.putText(frame, text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.1)

@app.route('/video_feed/<camera_ip>')
def video_feed(camera_ip):
    return Response(generate_frames_for_camera(camera_ip),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_active_cameras')
def get_active_cameras():
    with camera_ips_lock:
        # 최근 30초 이내에 통신한 카메라만 반환
        current_time = datetime.now()
        active_cameras = {ip: last_seen for ip, last_seen in camera_ips.items()
                         if (current_time - last_seen).seconds < 30}
    return jsonify(list(active_cameras.keys()))

@app.route('/latest_results')
def get_latest_results():
    with results_lock:
        if latest_results is None:
            return jsonify({"error": "아직 처리된 결과가 없습니다."})
        return jsonify(latest_results)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_file(f'static/{filename}')

@app.route('/events/<path:filename>')
def serve_image(filename):
    return send_file(f'events/{filename}', mimetype='image/jpeg')

if __name__ == '__main__':
    os.makedirs("events", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
from flask import Flask, request, jsonify, render_template, Response, send_file, redirect, session, url_for, stream_with_context
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
import sqlite3
from pathlib import Path
from functools import wraps

app = Flask(__name__)
app.secret_key = 'secret_key'  # 세션 관리를 위한 비밀 키 설정

# YOLOv8 모델 로드
person_model = YOLO('best_person.pt').to('cpu')  # 사람 인식용 경량 모델 사용
cars_model = YOLO('best_cars.pt').to('cpu')  # 차량 인식용 경량 모델 사용
total_model = YOLO('yolov8n.pt').to('cpu')  # 기본 모델 로드

# 초기 빈 리스트로 설정
p_ip = [] # 사람 인식용 IP 리스트
v_ip = [] # 차량 인식용 IP 리스트

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

# 데이터베이스 초기화 함수 수정
def init_db():
    db_path = Path(__file__).parent / 'cameras.db'
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    # 사용자 테이블 생성
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')

    # 초기 관리자 계정 추가 (이미 없으면)
    c.execute('SELECT COUNT(*) FROM users WHERE username = "admin"')
    if c.fetchone()[0] == 0:
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', ('admin', '123456'))

    # 카메라 테이블 생성
    c.execute('''
        CREATE TABLE IF NOT EXISTS cameras
        (ip TEXT PRIMARY KEY, type TEXT NOT NULL)
    ''')
    
    # 이벤트 테이블 생성
    c.execute('''
        CREATE TABLE IF NOT EXISTS events
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         target_type TEXT NOT NULL,
         camera_ip TEXT NOT NULL,
         detected_at DATETIME NOT NULL,
         image_path TEXT,
         object_count INTEGER NOT NULL,
         FOREIGN KEY (camera_ip) REFERENCES cameras(ip))
    ''')
    
    # 대시보드 통계 테이블 생성
    c.execute('''
        CREATE TABLE IF NOT EXISTS dashboard_stats
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         camera_ip TEXT NOT NULL,
         today_person_count INTEGER DEFAULT 0,
         today_vehicle_count INTEGER DEFAULT 0,
         today_event_count INTEGER DEFAULT 0,
         last_updated DATETIME NOT NULL,
         FOREIGN KEY (camera_ip) REFERENCES cameras(ip))
    ''')

    # 초기 카메라 데이터가 없는 경우에만 기본값 추가
    c.execute('SELECT COUNT(*) FROM cameras')
    if c.fetchone()[0] == 0:
        c.execute('INSERT INTO cameras (ip, type) VALUES (?, ?)', ('192.168.50.16', 'person'))
        c.execute('INSERT INTO cameras (ip, type) VALUES (?, ?)', ('192.168.50.17', 'vehicle'))
    
    # 대시보드 통계 초기화
    c.execute('DELETE FROM dashboard_stats')  # 기존 통계 삭제
    for camera in get_cameras():
        c.execute('''
            INSERT INTO dashboard_stats 
            (camera_ip, today_person_count, today_vehicle_count, today_event_count, last_updated)
            VALUES (?, 0, 0, 0, datetime('now', 'localtime'))
        ''', (camera['ip'],))
    
    conn.commit()
    conn.close()
    
    # DB에서 카메라 리스트 초기화
    update_camera_lists()

# 이벤트 저장 함수
def save_event(target_type, camera_ip, image_path, object_count):
    db_path = Path(__file__).parent / 'cameras.db'
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO events (target_type, camera_ip, detected_at, image_path, object_count)
        VALUES (?, ?, datetime('now', 'localtime'), ?, ?)
    ''', (target_type, camera_ip, image_path, object_count))
    
    conn.commit()
    conn.close()
    
    # 대시보드에 이벤트 발생 알림 (SSE)
    notify_dashboard_event()

# 이벤트 조회 함수
def get_events_page(page=1, per_page=10):
    db_path = Path(__file__).parent / 'cameras.db'
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    
    # 전체 이벤트 수 조회
    c.execute('SELECT COUNT(*) FROM events')
    total_count = c.fetchone()[0]
    
    # 페이지네이션된 이벤트 조회
    offset = (page - 1) * per_page
    c.execute('''
        SELECT id, target_type, camera_ip, detected_at, image_path, object_count 
        FROM events 
        ORDER BY detected_at DESC 
        LIMIT ? OFFSET ?
    ''', (per_page, offset))
    
    events = [{
        'id': row[0],
        'target_type': row[1],
        'camera_ip': row[2],
        'detected_at': row[3],
        'image_path': row[4],
        'object_count': row[5]
    } for row in c.fetchall()]
    
    conn.close()
    
    total_pages = (total_count + per_page - 1) // per_page
    return events, total_pages

# 데이터베이스에서 카메라 목록 가져오기
def get_cameras():
    db_path = Path(__file__).parent / 'cameras.db'
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute('SELECT * FROM cameras')
    cameras = [{'ip': row[0], 'type': row[1]} for row in c.fetchall()]
    conn.close()
    return cameras

# 카메라 IP 리스트 업데이트
def update_camera_lists():
    global p_ip, v_ip
    cameras = get_cameras()
    p_ip = [cam['ip'] for cam in cameras if cam['type'] == 'person']
    v_ip = [cam['ip'] for cam in cameras if cam['type'] == 'vehicle']

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
        os.makedirs(events_dir, exist_ok=True)  
        image_path = os.path.join(events_dir, event_image_filename)
        cv2.imwrite(image_path, annotated_img)  
        latest_results["image_url"] = f"/image/{event_image_filename}"  
        
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
        cv2.putText(image, text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

@app.route('/detect', methods=['POST'])
def detect():
    global camera_ip
    global p_ip, v_ip
    
    try:
        camera_ip = request.args.get('ip')
        print(f"카메라 IP from URL: {camera_ip}")

        img_data = request.data
        if not img_data:
            return jsonify({"error": "이미지 데이터가 없습니다.", "danger": False}), 400

        if camera_ip:
            with camera_ips_lock:
                camera_ips[camera_ip] = datetime.now()

        # 바이너리 이미지 데이터를 직접 디코딩
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "이미지를 처리할 수 없습니다.", "danger": False}), 400
                
        img = cv2.flip(img, -1)
        img = cv2.resize(img, (320, 240))

        with image_lock:
            latest_images[camera_ip] = img.copy()

        # 위험 감지
        result, annotated_img = detect_danger(img)

        # 위험한 경우 이벤트 기록 및 LED 알림
        if result["danger"]:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # 이미지 저장
            path = os.path.dirname(os.path.abspath(__file__))
            filename = f"/events/event_{timestamp.replace(' ', '_').replace(':', '-')}.jpg"
            full_path = path + filename
            os.makedirs(path + "/events", exist_ok=True)
            cv2.imwrite(full_path, annotated_img)

            # LED 알림 설정
            if camera_ip in p_ip:  # 보행자 카메라에서 감지
                arduino_url = "http://" + v_ip[0]  # 차량용 LED
                target_type = "person"
                object_count = result["persons"]
            elif camera_ip in v_ip:  # 차량 카메라에서 감지
                arduino_url = "http://" + p_ip[0]  # 보행자용 LED
                target_type = "vehicle"
                object_count = result["vehicles"]
            
            print(f"Alert from {camera_ip} to Arduino at: {arduino_url}")
            
            try:
                # LED 제어 요청
                params = {
                    'danger': 1,
                    'person': result["persons"],
                    'vehicle': result["vehicles"],
                    'message': result["message"]
                }
                requests.get(arduino_url, params=params, timeout=0.1)
            except requests.exceptions.RequestException as e:
                print(f"Failed to send alert to Arduino: {e}")
            
            # 이벤트 저장
            save_event(target_type, camera_ip, filename, object_count)
            
            # 이벤트 목록 업데이트
            with events_lock:
                events.append({
                    "timestamp": timestamp,
                    "message": result["message"],
                    "image": filename
                })
                if len(events) > 100:
                    events.pop(0)

        return jsonify({
            "danger": result["danger"],
            "message": result["message"],
            "persons": result["persons"],
            "vehicles": result["vehicles"]
        })

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e), "danger": False}), 400

@app.route('/events', methods=['GET'])
def get_events():
    with events_lock:
        
        for event in events:
            if "image" in event:
                event["image_url"] = f"/events/{os.path.basename(event['image'])}"
        return jsonify(events)

# 카메라별 비디오 스트림 생성
def draw_bounding_boxes(frame, objects, camera_ip, p_ip, v_ip):
    """객체 정보를 이용하여 바운딩 박스와 라벨을 그리는 함수"""
    for obj in objects:
        label = obj["type"]
        confidence = obj["confidence"]
        bbox = obj["bbox"]
        x1, y1, x2, y2 = bbox

        # 객체 유형에 따라 색상 설정
        is_person = camera_ip in p_ip and label == "person"
        is_vehicle = camera_ip in v_ip and label in ["bicycle", "car", "motorcycle", "bus", "truck"]

        if is_person:
            color = (0, 255, 0)  # 초록색 (사람)
        elif is_vehicle:
            color = (0, 0, 255)  # 빨간색 (차량)
        else:
            color = (255, 255, 0)  # 노란색 (기타)

        # 바운딩 박스와 라벨 그리기
        if is_person or is_vehicle:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {confidence:.2f}"
            cv2.putText(frame, text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
                        draw_bounding_boxes(frame, latest_results["objects"], camera_ip, p_ip, v_ip)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.1)

@app.route('/video_feed/<camera_ip>')
def video_feed(camera_ip):
    print(f"Video feed requested for camera IP: {camera_ip}")
    return Response(generate_frames_for_camera(camera_ip),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_active_cameras')
def get_active_cameras():
    cameras = get_cameras()  # 데이터베이스에서 카메라 목록 가져오기
    active_cameras = []
    with camera_ips_lock:
        # 최근 30초 이내에 통신한 카메라만 활성 카메라로 간주
        current_time = datetime.now()
        active_cameras = [
            cam['ip'] for cam in cameras
            if cam['ip'] in camera_ips and (current_time - camera_ips[cam['ip']]).seconds < 30
        ]
    print(f"Active cameras: {active_cameras}")
    return jsonify(active_cameras)

@app.route('/latest_results')
def get_latest_results():
    with results_lock:
        if latest_results is None:
            return jsonify({"error": "아직 처리된 결과가 없습니다."})
        return jsonify(latest_results)

# 로그인 데코레이터
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('username') is None:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# 로그인 라우트
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        db_path = Path(__file__).parent / 'cameras.db'
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

# 로그아웃 라우트
@app.route('/logout')
def logout():
    session.clear()  # 세션 초기화
    return redirect(url_for('login'))

# index 페이지 접근 시 로그인 확인
@app.route('/')
@login_required
def index():
    return render_template('index.html')

# dashboard 페이지 접근 시 로그인 확인
@app.route('/dashboard')
@login_required
def dashboard():
    cameras = get_cameras()  # 카메라 목록 가져오기
    stats = get_dashboard_data()  # 통계 데이터 가져오기
    
    # 이벤트 데이터 가져오기
    db_path = Path(__file__).parent / 'cameras.db'
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute("SELECT target_type, camera_ip, detected_at FROM events ORDER BY detected_at DESC LIMIT 10")
    events = [{'target_type': row[0], 'camera_ip': row[1], 'detected_at': row[2]} for row in c.fetchall()]
    conn.close()
    
    return render_template('dashboard.html', stats=stats, cameras=cameras, events=events)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_file(f'static/{filename}')

@app.route('/events/<path:filename>')
def serve_image(filename):
    return send_file(f'events/{filename}', mimetype='image/jpeg')

@app.route('/camera_register')
def camera_register():
    cameras = get_cameras()
    return render_template('camera_register.html', cameras=cameras)

@app.route('/register_camera', methods=['POST'])
def register_camera():
    try:
        camera_ip = request.form['camera_ip']
        camera_type = request.form['camera_type']
        
        db_path = Path(__file__).parent / 'cameras.db'
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.execute('INSERT OR REPLACE INTO cameras (ip, type) VALUES (?, ?)', 
                 (camera_ip, camera_type))
        conn.commit()
        conn.close()
        
        update_camera_lists()
        return redirect('/camera_register')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/delete_camera', methods=['POST'])
def delete_camera():
    try:
        data = request.get_json()
        camera_ip = data['camera_ip']
        
        db_path = Path(__file__).parent / 'cameras.db'
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.execute('DELETE FROM cameras WHERE ip = ?', (camera_ip,))
        conn.commit()
        conn.close()
        
        update_camera_lists()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/events')
def view_events():
    page = request.args.get('page', 1, type=int)
    events_list, total_pages = get_events_page(page)
    return render_template('events.html', 
                         events=events_list, 
                         current_page=page,
                         total_pages=total_pages)

# 대시보드 통계 업데이트 함수
def update_dashboard_stats():
    db_path = Path(__file__).parent / 'cameras.db'
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    
    # 오늘 날짜 기준으로 통계 계산
    today = datetime.now().strftime('%Y-%m-%d')
    
    c.execute('''
        INSERT OR REPLACE INTO dashboard_stats 
        (camera_ip, today_person_count, today_vehicle_count, today_event_count, last_updated)
        SELECT 
            camera_ip,
            SUM(CASE WHEN target_type = 'person' THEN object_count ELSE 0 END) as person_count,
            SUM(CASE WHEN target_type = 'vehicle' THEN object_count ELSE 0 END) as vehicle_count,
            COUNT(*) as event_count,
            datetime('now', 'localtime')
        FROM events
        WHERE date(detected_at) = ?
        GROUP BY camera_ip
    ''', (today,))
        
    conn.commit()
    conn.close()

# 대시보드 데이터 조회 함수
def get_dashboard_data():
    db_path = Path(__file__).parent / 'cameras.db'
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    
    # 최근 대시보드 통계 조회
    c.execute('''
        SELECT camera_ip, today_person_count, today_vehicle_count, 
               today_event_count, last_updated
        FROM dashboard_stats
        ORDER BY last_updated DESC
    ''')
    
    stats = [{
        'camera_ip': row[0],
        'person_count': row[1],
        'vehicle_count': row[2],
        'event_count': row[3],
        'last_updated': row[4]
    } for row in c.fetchall()]
        
    conn.close()
    return stats

# SSE 엔드포인트
@app.route('/dashboard_events')
def stream_dashboard_events():
    def event_stream():
        while True:
            # 새로운 이벤트가 발생할 때까지 대기
            time.sleep(1)
            
            # 대시보드 데이터 가져오기
            stats = get_dashboard_data()
            events = get_recent_events()
            
            # 이벤트 데이터 생성 (JSON 형식)
            event_data = json.dumps({'stats': stats, 'events': events})
            
            # SSE 형식으로 데이터 전송
            yield f"data: {event_data}\n\n"
    
    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')

# 최근 이벤트 데이터 조회 함수
def get_recent_events():
    db_path = Path(__file__).parent / 'cameras.db'
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute("SELECT target_type, camera_ip, detected_at FROM events ORDER BY detected_at DESC LIMIT 10")
    events = [{'target_type': row[0], 'camera_ip': row[1], 'detected_at': row[2]} for row in c.fetchall()]
    conn.close()
    return events

if __name__ == '__main__':
    init_db()  # 데이터베이스 초기화
    update_camera_lists()  # 카메라 목록 업데이트
    os.makedirs("events", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
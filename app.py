import cv2
import time
import threading
from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import numpy as np
from pushbullet import Pushbullet

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ================= PUSHBULLET CONFIGURATION =================
# !!! REPLACE THIS WITH YOUR ACTUAL API KEY FROM pushbullet.com/account !!!
PUSHBULLET_API_KEY = "o.G0feXnHZD9PvUXR3iDqgqewfFnMJ3w12"  # <-- CHANGE THIS
pb = None

def init_pushbullet():
    """Initialize Pushbullet connection"""
    global pb
    try:
        pb = Pushbullet(PUSHBULLET_API_KEY)
        print(f"✅ Pushbullet ready - Sending to {len(pb.devices)} device(s)")
        for device in pb.devices:
            print(f"   📱 {device.nickname}")
        return True
    except Exception as e:
        print(f"⚠️ Pushbullet not configured: {e}")
        print("   Get API key from: https://www.pushbullet.com/account")
        return False

def send_anomaly_notification(activity_level, person_count, cause):
    """Send anomaly alert to your phone via Pushbullet"""
    global pb
    if not pb and not init_pushbullet():
        return False
    
    title = "⚠️ HEAT RISK ANOMALY!"
    message = f"Cause: {cause}\nActivity: {activity_level:.3f}\nPeople: {person_count}\nTime: {time.strftime('%H:%M:%S')}"
    
    try:
        pb.push_note(title, message)
        print(f"📱 Push notification sent: {title}")
        return True
    except Exception as e:
        print(f"❌ Notification failed: {e}")
        return False

def send_test_notification():
    """Send a test notification to verify setup"""
    global pb
    if not pb and not init_pushbullet():
        return False
    
    try:
        pb.push_note("✅ AI Smart Home Test", 
                     f"System is running!\nTime: {time.strftime('%H:%M:%S')}\nYour phone is connected.")
        print("📱 Test notification sent successfully!")
        return True
    except Exception as e:
        print(f"❌ Test notification failed: {e}")
        return False

# ================= CONFIGURATION =================
OFF_DELAY = 5
VELOCITY_THRESH = 0.02
HIGH_ACTIVITY_THRESH = 0.045

# ================= GLOBAL VARIABLES =================
camera_active = False
cap = None
model = None
previous_centroids = []
last_human_time = time.time()
last_sent = None
last_notification_time = 0
NOTIFICATION_COOLDOWN = 30  # Seconds between notifications

system_state = {
    "human_detected": False,
    "person_count": 0,
    "activity_level": 0.0,
    "activity_label": "No person",
    "esp_ip": "192.168.1.100",
    "esp_connected": False,
    "camera_running": False
}

# ================= LOAD YOLO =================
def load_model():
    global model
    if model is None:
        print("Loading YOLO model...")
        model = YOLO("yolov8n.pt")
        print("✅ YOLO model loaded")
    return model

# ================= ACTIVITY DETECTION =================
def calculate_movement_activity(detections):
    global previous_centroids
    
    current_centroids = []
    for (x1, y1, x2, y2) in detections:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        current_centroids.append((cx, cy))
    
    person_count = len(current_centroids)
    
    if previous_centroids and len(previous_centroids) == person_count and person_count > 0:
        movements = []
        for i, (cx, cy) in enumerate(current_centroids):
            if i < len(previous_centroids):
                px, py = previous_centroids[i]
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                movements.append(dist)
        
        if movements:
            avg_movement = np.mean(movements)
            activity = min(0.1, avg_movement / 500)
        else:
            activity = 0
    else:
        activity = 0
    
    previous_centroids = current_centroids
    return activity, person_count

# ================= ESP32 COMMAND =================
def send_esp32_command(command):
    """Send command to ESP32 (simulated for now)"""
    esp_ip = system_state.get("esp_ip", "Not set")
    print(f"📡 [SIMULATED] Sending {command} to ESP32 at {esp_ip}")
    
    if esp_ip and esp_ip != "Not set":
        system_state["esp_connected"] = True
        return True
    else:
        system_state["esp_connected"] = False
        return False

# ================= VIDEO GENERATOR =================
def generate_frames():
    global camera_active, cap, last_human_time, last_sent, previous_centroids, last_notification_time
    
    load_model()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    camera_active = True
    system_state["camera_running"] = True
    
    print("📹 Camera streaming started")
    
    while camera_active:
        success, frame = cap.read()
        if not success:
            break
        
        # YOLO detection
        results = model(frame, imgsz=320, conf=0.55, verbose=False)
        
        person_boxes = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "PERSON", (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        person_count = len(person_boxes)
        activity, _ = calculate_movement_activity(person_boxes)
        
        # Update state
        system_state["person_count"] = person_count
        system_state["activity_level"] = activity
        
        is_anomaly = False
        anomaly_cause = ""
        
        if activity > HIGH_ACTIVITY_THRESH:
            system_state["activity_label"] = "🔥 High activity (heat risk)"
            is_anomaly = True
            anomaly_cause = "High activity - Vigorous movement detected!"
        elif activity > VELOCITY_THRESH:
            system_state["activity_label"] = "⚡ Medium activity"
        elif person_count > 0:
            system_state["activity_label"] = "💚 Low activity"
        else:
            system_state["activity_label"] = "😴 No person"
        
        # Send PUSHBULLET NOTIFICATION for anomaly
        current_time = time.time()
        if is_anomaly and (current_time - last_notification_time) > NOTIFICATION_COOLDOWN:
            send_anomaly_notification(activity, person_count, anomaly_cause)
            last_notification_time = current_time
            print(f"🔔 ANOMALY NOTIFICATION SENT! Activity: {activity:.3f}")
        
        # ESP32 control based on human presence
        human_present = person_count > 0
        system_state["human_detected"] = human_present
        
        if human_present:
            last_human_time = time.time()
            if last_sent != "ON":
                send_esp32_command("ON")
                last_sent = "ON"
                socketio.emit('state_update', system_state)
        else:
            if time.time() - last_human_time > OFF_DELAY:
                if last_sent != "OFF":
                    send_esp32_command("OFF")
                    last_sent = "OFF"
                    socketio.emit('state_update', system_state)
        
        # Draw on frame
        status_color = (0, 255, 0) if human_present else (0, 0, 255)
        cv2.putText(frame, f"People: {person_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Activity: {activity:.3f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        if is_anomaly:
            cv2.putText(frame, "⚠️ ANOMALY!", (frame.shape[1]-150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        # Encode and yield frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        socketio.emit('state_update', system_state)
        time.sleep(0.05)
    
    cap.release()
    camera_active = False
    system_state["camera_running"] = False
    print("📹 Camera streaming stopped")

# ================= FLASK ROUTES =================
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state')
def get_state():
    return jsonify(system_state)

@app.route('/api/set_esp_ip', methods=['POST'])
def set_esp_ip():
    data = request.json
    system_state["esp_ip"] = data.get("ip", "Not set")
    print(f"📡 ESP32 IP updated to: {system_state['esp_ip']}")
    return jsonify({"success": True, "ip": system_state["esp_ip"]})

@app.route('/api/test_esp')
def test_esp():
    esp_ip = system_state.get("esp_ip", "Not set")
    if esp_ip and esp_ip != "Not set":
        system_state["esp_connected"] = True
        return jsonify({"connected": True, "ip": esp_ip, "message": f"ESP32 at {esp_ip} is reachable (simulated)"})
    else:
        system_state["esp_connected"] = False
        return jsonify({"connected": False, "message": "No ESP32 IP configured"})

@app.route('/api/test_notification')
def test_notification():
    """Test endpoint to verify push notifications are working"""
    result = send_test_notification()
    return jsonify({"success": result, "message": "Test notification sent" if result else "Failed to send"})

@app.route('/api/stop_camera')
def stop_camera():
    global camera_active
    camera_active = False
    system_state["camera_running"] = False
    return jsonify({"success": True})

@socketio.on('connect')
def handle_connect():
    print("🌐 Web client connected")
    emit('state_update', system_state)

# ================= MAIN =================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 AI SMART HOME DASHBOARD")
    print("="*50)
    
    # Initialize Pushbullet
    init_pushbullet()
    
    print("\n📱 Pushbullet notifications are ACTIVE")
    print("📹 Open browser to: http://localhost:5000")
    print("⚙️  Enter your ESP32 IP in the dashboard")
    print("🔔 High activity will trigger phone notifications")
    print("📲 Test notification available in dashboard")
    print("="*50 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
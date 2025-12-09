import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import json
import asyncio
import websockets
from threading import Thread
import time
import socket


screen_w, screen_h = pyautogui.size()

WS_HOST = "localhost"  # Listen on all interfaces
WS_PORT = 8765

print(f"WebSocket target port: {WS_PORT}")

def get_local_ip_addresses():
    """Collect available IPv4 addresses for user instructions."""
    ips = set()
    try:
        hostname = socket.gethostname()
        host_ip = socket.gethostbyname(hostname)
        if host_ip:
            ips.add(host_ip)
        for info in socket.getaddrinfo(hostname, None):
            ip = info[4][0]
            if ":" not in ip:
                ips.add(ip)
    except Exception:
        pass
    ips.add("127.0.0.1")
    return sorted(ips)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

smooth_x, smooth_y = screen_w // 2, screen_h // 2
alpha = 0.15  

class KalmanFilter:
    def __init__(self, process_variance=1e-3, measurement_variance=1e-1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_x = screen_w // 2
        self.estimated_y = screen_h // 2
        self.error_cov_x = 1.0
        self.error_cov_y = 1.0
    
    def update(self, measurement_x, measurement_y):
        prediction_x = self.estimated_x
        prediction_y = self.estimated_y
        error_cov_x = self.error_cov_x + self.process_variance
        error_cov_y = self.error_cov_y + self.process_variance
        
        kalman_gain_x = error_cov_x / (error_cov_x + self.measurement_variance)
        kalman_gain_y = error_cov_y / (error_cov_y + self.measurement_variance)
        
        self.estimated_x = prediction_x + kalman_gain_x * (measurement_x - prediction_x)
        self.estimated_y = prediction_y + kalman_gain_y * (measurement_y - prediction_y)
        
        self.error_cov_x = (1 - kalman_gain_x) * error_cov_x
        self.error_cov_y = (1 - kalman_gain_y) * error_cov_y
        
        return int(self.estimated_x), int(self.estimated_y)

kalman_filter = KalmanFilter(process_variance=1e-4, measurement_variance=5e-2)  

dot_window = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
cv2.namedWindow('Gaze Pointer', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Gaze Pointer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty('Gaze Pointer', cv2.WND_PROP_TOPMOST, 1)

calibration_active = True
calibration_start_time = None
calibration_duration = 5.0 
calibration_samples = [] 
gaze_vector_offset = np.array([0.0, 0.0, 0.0])  

def estimate_head_pose(landmarks, frame_w, frame_h):
    model_points = np.array([
        (0.0, 0.0, 0.0),            
        (0.0, -330.0, -65.0),      
        (-225.0, 170.0, -135.0),   
        (225.0, 170.0, -135.0),    
        (-150.0, -150.0, -125.0),   
        (150.0, -150.0, -125.0)     
    ], dtype=np.float64)
    
    image_points = np.array([
        (landmarks[1].x * frame_w, landmarks[1].y * frame_h),      # Nose
        (landmarks[152].x * frame_w, landmarks[152].y * frame_h),  # Chin
        (landmarks[33].x * frame_w, landmarks[33].y * frame_h),    # Left eye
        (landmarks[263].x * frame_w, landmarks[263].y * frame_h),  # Right eye
        (landmarks[61].x * frame_w, landmarks[61].y * frame_h),    # Left mouth
        (landmarks[291].x * frame_w, landmarks[291].y * frame_h)   # Right mouth
    ], dtype=np.float64)
    
    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.zeros((4, 1))
    
    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    return rotation_vec, translation_vec, camera_matrix

def get_iris_position_3d(landmarks, eye_indices, iris_indices):
    iris_center = np.mean([(landmarks[idx].x, landmarks[idx].y, landmarks[idx].z) 
                           for idx in iris_indices], axis=0)
    
    eye_center = np.mean([(landmarks[idx].x, landmarks[idx].y, landmarks[idx].z) 
                          for idx in eye_indices], axis=0)
    
    return iris_center, eye_center

def calculate_gaze_direction(landmarks, frame_w, frame_h):
    rotation_vec, translation_vec, camera_matrix = estimate_head_pose(landmarks, frame_w, frame_h)
    
    left_iris, left_eye = get_iris_position_3d(landmarks, LEFT_EYE, LEFT_IRIS)
    right_iris, right_eye = get_iris_position_3d(landmarks, RIGHT_EYE, RIGHT_IRIS)
    
    left_gaze = np.array(left_iris) - np.array(left_eye)
    right_gaze = np.array(right_iris) - np.array(right_eye)
    
    avg_gaze = (left_gaze + right_gaze) / 2
    
    distance = abs(translation_vec[2][0]) / 1000.0  
    
    return avg_gaze, rotation_vec, distance

def project_gaze_to_screen(gaze_vector, rotation_vec, distance, frame_w, frame_h, sensitivity_multiplier=1.0, gaze_offset=None):
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    
    gaze_world = rotation_mat @ gaze_vector
    
    # Apply calibration offset if available
    if gaze_offset is not None:
        gaze_world = gaze_world - gaze_offset
    
    monitor_distance = 0.6  
    scale_factor = distance / monitor_distance
    
    # Base sensitivities
    base_sensitivity_x = 25.5
    base_sensitivity_y = 50.0  # Y-axis
    
    sensitivity_x = base_sensitivity_x * sensitivity_multiplier
    sensitivity_y = base_sensitivity_y * sensitivity_multiplier
    
    screen_x = screen_w / 2 + gaze_world[0] * screen_w * sensitivity_x * scale_factor
    screen_y = screen_h / 2 - gaze_world[1] * screen_h * sensitivity_y * scale_factor
    return int(screen_x), int(screen_y), gaze_world

def get_eye_position(landmarks, eye_indices, iris_indices, frame_w, frame_h):
    iris_center = np.mean([(landmarks[idx].x, landmarks[idx].y) 
                           for idx in iris_indices], axis=0)
    
    eye_points = [(landmarks[idx].x, landmarks[idx].y) 
                  for idx in eye_indices]
    
    eye_left = min([p[0] for p in eye_points])
    eye_right = max([p[0] for p in eye_points])
    eye_top = min([p[1] for p in eye_points])
    eye_bottom = max([p[1] for p in eye_points])
    
    eye_width = eye_right - eye_left
    eye_height = eye_bottom - eye_top
    
    if eye_width > 0 and eye_height > 0:
        x_ratio = (iris_center[0] - eye_left) / eye_width
        y_ratio = (iris_center[1] - eye_top) / eye_height
        return x_ratio, y_ratio
    
    return 0.5, 0.5

print("=" * 60)
print("Eye Gaze Tracker - 3D Отслеживание взгляда")
print("=" * 60)
print(f"Размер экрана: {screen_w}x{screen_h}")
accessible_urls = ", ".join([f"ws://{ip}:{WS_PORT}" for ip in get_local_ip_addresses()])
print(f"WebSocket URLs: {accessible_urls}")
print("\n=== CALIBRATION PHASE ===")
print("Please look at the CENTER of the screen for 5 seconds...")
print("The red dot will be locked to center during calibration.")
print("=" * 60)
print("\nAfter calibration:")
print("- Смотрите на камеру и двигайте взглядом")
print("- Красная точка показывает направление вашего взгляда")
print("- Нажмите 'q' для выхода")
print("- Нажмите 'c' для повторной калибровки центра")
print("- Нажмите '+/-' для изменения чувствительности")
print("=" * 60)

calibration_start_time = time.time()
calibration_samples = []

offset_x, offset_y = 0, 0
sensitivity_multiplier = 1.0

gaze_data_clients = set()
websocket_server_ready = False

latest_gaze_data = {
    "x": 0.5,
    "y": 0.5,
    "screenX": screen_w // 2,
    "screenY": screen_h // 2,
    "confidence": 0.0,
    "distance": 0.0,
    "timestamp": 0.0
}

async def register_client(websocket):
    gaze_data_clients.add(websocket)
    print(f"[✓] WebSocket client connected. Total clients: {len(gaze_data_clients)}")
    try:
        await websocket.wait_closed()
    finally:
        gaze_data_clients.remove(websocket)
        print(f"[!] WebSocket client disconnected. Remaining clients: {len(gaze_data_clients)}")

async def broadcast_gaze_data():
    """Continuously broadcast the latest gaze data to all connected clients"""
    while True:
        if gaze_data_clients:
            message = json.dumps(latest_gaze_data)
            disconnected = set()
            for client in gaze_data_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
                except Exception as e:
                    print(f"[!] Error sending to client: {e}")
                    disconnected.add(client)
            gaze_data_clients.difference_update(disconnected)
        await asyncio.sleep(0.016)  # ~60fps

async def run_server():
    global websocket_server_ready
    try:
        async with websockets.serve(register_client, WS_HOST, WS_PORT):
            websocket_server_ready = True
            print(f"[✓] WebSocket server listening on port {WS_PORT}")
            for ip in get_local_ip_addresses():
                print(f"    • ws://{ip}:{WS_PORT}")
            await broadcast_gaze_data()
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"[!] WebSocket port {WS_PORT} is already in use. Is another instance running?")
        else:
            print(f"[!] WebSocket server error: {e}")
        websocket_server_ready = False
    except Exception as e:
        print(f"[!] WebSocket server error: {e}")
        websocket_server_ready = False

def start_websocket_server():
    """Start WebSocket server in a separate thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_server())
    except Exception as e:
        print(f"[!] Failed to start WebSocket server: {e}")

ws_thread = Thread(target=start_websocket_server, daemon=True)
ws_thread.start()

time.sleep(1.0)
if websocket_server_ready:
    print("[✓] WebSocket server is ready and accepting connections")
else:
    print("[!] WebSocket server may not be ready yet.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Не удалось получить кадр с камеры")
        break
    
    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    current_time = time.time()
    if calibration_active:
        elapsed = current_time - calibration_start_time
        remaining = calibration_duration - elapsed
        calibration_in_progress = remaining > 0
        if not calibration_in_progress:
            calibration_active = False
            if len(calibration_samples) > 0:
                samples_array = np.array(calibration_samples)
                gaze_vector_offset = np.mean(samples_array, axis=0)
                print(f"\n[✓] Calibration complete! Gaze offset: {gaze_vector_offset}")
            calibration_in_progress = False
    else:
        calibration_in_progress = False
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark
        try:
            gaze_vector, rotation_vec, distance = calculate_gaze_direction(landmarks, frame_w, frame_h)
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            gaze_world = rotation_mat @ gaze_vector
            
            if calibration_in_progress:
                calibration_samples.append(gaze_world.copy())
                smooth_x = screen_w // 2
                smooth_y = screen_h // 2
            else:
                gaze_x, gaze_y, gaze_world_corrected = project_gaze_to_screen(
                    gaze_vector, rotation_vec, distance, frame_w, frame_h, 
                    sensitivity_multiplier, gaze_vector_offset
                )
                gaze_x += offset_x
                gaze_y += offset_y
                gaze_x = max(0, min(screen_w - 1, gaze_x))
                gaze_y = max(0, min(screen_h - 1, gaze_y))
                smooth_x = int(alpha * gaze_x + (1 - alpha) * smooth_x)
                smooth_y = int(alpha * gaze_y + (1 - alpha) * smooth_y)
            
            normalized_x = smooth_x / screen_w
            normalized_y = 1.0 - (smooth_y / screen_h)
            
            gaze_magnitude = np.linalg.norm(gaze_vector)
            confidence = min(1.0, max(0.0, 1.0 - abs(distance - 0.6) / 0.3)) * min(1.0, gaze_magnitude * 10)
            
            # Update latest gaze data for WebSocket broadcast
            if not calibration_in_progress:
                latest_gaze_data = {
                    "x": float(normalized_x),
                    "y": float(normalized_y),
                    "screenX": int(smooth_x),
                    "screenY": int(smooth_y),
                    "confidence": float(confidence),
                    "distance": float(distance),
                    "timestamp": float(cv2.getTickCount() / cv2.getTickFrequency())
                }
            
            dot_window.fill(0)
            if calibration_in_progress:
                cv2.circle(dot_window, (smooth_x, smooth_y), 25, (0, 255, 255), -1)
                cv2.circle(dot_window, (smooth_x, smooth_y), 27, (255, 255, 255), 3)
                progress = (current_time - calibration_start_time) / calibration_duration
                angle = int(360 * progress)
                cv2.ellipse(dot_window, (smooth_x, smooth_y), (40, 40), 0, 0, angle, (0, 255, 0), 3)
            else:
                cv2.circle(dot_window, (smooth_x, smooth_y), 20, (0, 0, 255), -1)
                cv2.circle(dot_window, (smooth_x, smooth_y), 22, (255, 255, 255), 2)
                
            cv2.imshow('Gaze Pointer', dot_window)
            
            for idx in LEFT_IRIS + RIGHT_IRIS:
                x = int(landmarks[idx].x * frame_w)
                y = int(landmarks[idx].y * frame_h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
        except Exception as e:
            left_x, left_y = get_eye_position(landmarks, LEFT_EYE, LEFT_IRIS, frame_w, frame_h)
            right_x, right_y = get_eye_position(landmarks, RIGHT_EYE, RIGHT_IRIS, frame_w, frame_h)
            avg_x = (left_x + right_x) / 2
            avg_y = (left_y + right_y) / 2
            gaze_x = int((1 - avg_x) * screen_w) + offset_x
            gaze_y = int(avg_y * screen_h) + offset_y
            gaze_x = max(0, min(screen_w - 1, gaze_x))
            gaze_y = max(0, min(screen_h - 1, gaze_y))
            smooth_x = int(alpha * gaze_x + (1 - alpha) * smooth_x)
            smooth_y = int(alpha * gaze_y + (1 - alpha) * smooth_y)
            dot_window.fill(0)
            cv2.circle(dot_window, (smooth_x, smooth_y), 20, (0, 0, 255), -1)
            cv2.imshow('Gaze Pointer', dot_window)
    else:
        cv2.putText(frame, "Лицо не обнаружено", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow('Eye Gaze Tracker (q=выход)', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        calibration_active = True
        calibration_start_time = time.time()
        calibration_samples = []
        gaze_vector_offset = np.array([0.0, 0.0, 0.0])
        print("\n[↻] Recalibration started - please look at center for 5 seconds...")
    elif key == ord('+') or key == ord('='):
        sensitivity_multiplier += 0.1
        print(f"\n[↑] Чувствительность: {sensitivity_multiplier:.1f}x")
    elif key == ord('-') or key == ord('_'):
        sensitivity_multiplier = max(0.1, sensitivity_multiplier - 0.1)
        print(f"\n[↓] Чувствительность: {sensitivity_multiplier:.1f}x")

cap.release()
cv2.destroyAllWindows()
print("\n\nПрограмма завершена.")
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import json
import asyncio
import websockets
from threading import Thread
import time
import os
import socket

screen_w, screen_h = pyautogui.size()

WS_HOST = os.environ.get("GAZE_WS_HOST", "0.0.0.0")
WS_PORT = int(os.environ.get("GAZE_WS_PORT", "8765"))

print(f"WebSocket target host: {WS_HOST}:{WS_PORT}")

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
NOSE_TIP = 1
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263

smooth_x, smooth_y = screen_w // 2, screen_h // 2
alpha = 0.2  

dot_window = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
cv2.namedWindow('Gaze Pointer', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Gaze Pointer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty('Gaze Pointer', cv2.WND_PROP_TOPMOST, 1)

calibration_points = []
calibration_mode = False
calibration_target = None

# Auto-calibration at startup
calibration_active = True
calibration_start_time = None
calibration_duration = 5.0  # 5 seconds
calibration_samples = []  # Store gaze_world samples during calibration
gaze_vector_offset = np.array([0.0, 0.0, 0.0])  # Offset to apply after calibration

def estimate_head_pose(landmarks, frame_w, frame_h):
    
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Кончик носа
        (0.0, -330.0, -65.0),        # Подбородок
        (-225.0, 170.0, -135.0),     # Левый угол глаза
        (225.0, 170.0, -135.0),      # Правый угол глаза
        (-150.0, -150.0, -125.0),    # Левый угол рта
        (150.0, -150.0, -125.0)      # Правый угол рта
    ], dtype=np.float64)
    
    image_points = np.array([
        (landmarks[1].x * frame_w, landmarks[1].y * frame_h),      # Нос
        (landmarks[152].x * frame_w, landmarks[152].y * frame_h),  # Подбородок
        (landmarks[33].x * frame_w, landmarks[33].y * frame_h),    # Левый глаз
        (landmarks[263].x * frame_w, landmarks[263].y * frame_h),  # Правый глаз
        (landmarks[61].x * frame_w, landmarks[61].y * frame_h),    # Левый рот
        (landmarks[291].x * frame_w, landmarks[291].y * frame_h)   # Правый рот
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
    
    # Base sensitivities - Y-axis needs higher base sensitivity for vertical movement
    base_sensitivity_x = 25.5
    base_sensitivity_y = 50.0  # Increased from 20.5 for better Y-axis responsiveness
    
    # Apply user-adjustable multiplier
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
print(f"WebSocket target host: {WS_HOST}:{WS_PORT}")
accessible_urls = ", ".join([f"ws://{ip}:{WS_PORT}" for ip in get_local_ip_addresses()])
print(f"Connect render module to one of: {accessible_urls}")
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

# Initialize calibration
calibration_start_time = time.time()
calibration_samples = []

offset_x, offset_y = 0, 0
sensitivity_multiplier = 1.0

# WebSocket server for gaze data broadcasting
gaze_data_clients = set()
gaze_data_queue = []
websocket_server_ready = False

async def register_client(websocket, path):
    gaze_data_clients.add(websocket)
    print(f"[✓] WebSocket client connected. Total clients: {len(gaze_data_clients)}")
    try:
        await websocket.wait_closed()
    finally:
        gaze_data_clients.remove(websocket)
        print(f"[!] WebSocket client disconnected. Remaining clients: {len(gaze_data_clients)}")

async def broadcast_gaze_data():
    while True:
        if gaze_data_queue and gaze_data_clients:
            data = gaze_data_queue.pop(0)
            message = json.dumps(data)
            disconnected = set()
            for client in gaze_data_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
            gaze_data_clients -= disconnected
        await asyncio.sleep(0.01)  # ~100 FPS

def start_websocket_server():
    global websocket_server_ready
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def run_server():
        global websocket_server_ready
        try:
            async with websockets.serve(register_client, WS_HOST, WS_PORT):
                websocket_server_ready = True
                if WS_HOST == "0.0.0.0":
                    print(f"[✓] WebSocket server listening on all interfaces (port {WS_PORT})")
                    print("    Reachable URLs:")
                    for ip in get_local_ip_addresses():
                        print(f"    • ws://{ip}:{WS_PORT}")
                else:
                    print(f"[✓] WebSocket server listening on ws://{WS_HOST}:{WS_PORT}")
                # Run broadcast loop concurrently with server
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
    
    try:
        loop.run_until_complete(run_server())
    except Exception as e:
        print(f"[!] Failed to start WebSocket server: {e}")
        websocket_server_ready = False

# Start WebSocket server in background thread
ws_thread = Thread(target=start_websocket_server, daemon=True)
ws_thread.start()

# Give server a moment to start
time.sleep(1.0)  # Increased wait time for server to initialize
if websocket_server_ready:
    print("[✓] WebSocket server is ready and accepting connections")
else:
    print("[!] WebSocket server may not be ready yet.")
    print("    This is normal if starting for the first time.")
    print("    The server will be ready shortly...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Не удалось получить кадр с камеры")
        break
    
    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    # Check calibration status
    current_time = time.time()
    if calibration_active:
        elapsed = current_time - calibration_start_time
        remaining = calibration_duration - elapsed
        
        if remaining > 0:
            # Still calibrating - show countdown and lock gaze to center
            calibration_in_progress = True
        else:
            # Calibration complete
            calibration_active = False
            if len(calibration_samples) > 0:
                # Calculate average gaze_world offset when looking at center
                samples_array = np.array(calibration_samples)
                gaze_vector_offset = np.mean(samples_array, axis=0)
                print(f"\n[✓] Calibration complete!")
                print(f"    Collected {len(calibration_samples)} samples")
                print(f"    Gaze offset: X={gaze_vector_offset[0]:.4f}, Y={gaze_vector_offset[1]:.4f}, Z={gaze_vector_offset[2]:.4f}")
                print(f"    You can now move your gaze around!\n")
            else:
                print("\n[!] Calibration failed - no samples collected. Using default offset.\n")
            calibration_in_progress = False
    else:
        calibration_in_progress = False
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark
        
        try:
            gaze_vector, rotation_vec, distance = calculate_gaze_direction(landmarks, frame_w, frame_h)
            
            # Calculate gaze_world for debug display
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            gaze_world = rotation_mat @ gaze_vector
            
            # During calibration, collect samples and lock gaze to center
            if calibration_in_progress:
                # Collect calibration sample
                calibration_samples.append(gaze_world.copy())
                
                # Lock gaze to center during calibration
                smooth_x = screen_w // 2
                smooth_y = screen_h // 2
                gaze_x = screen_w // 2
                gaze_y = screen_h // 2
            else:
                # Normal tracking mode - apply calibration offset
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
            
            # Normalize coordinates for VR rendering (0-1 range)
            normalized_x = smooth_x / screen_w
            normalized_y = 1.0 - (smooth_y / screen_h)
            
            # Calculate confidence based on distance and gaze vector magnitude
            gaze_magnitude = np.linalg.norm(gaze_vector)
            confidence = min(1.0, max(0.0, 1.0 - abs(distance - 0.6) / 0.3)) * min(1.0, gaze_magnitude * 10)
            
            # Broadcast gaze data via WebSocket (only after calibration)
            if not calibration_in_progress:
                gaze_data = {
                    "x": normalized_x,
                    "y": normalized_y,
                    "screenX": smooth_x,
                    "screenY": smooth_y,
                    "confidence": confidence,
                    "distance": float(distance),
                    "timestamp": cv2.getTickCount() / cv2.getTickFrequency()
                }
                if len(gaze_data_queue) < 10:  # Limit queue size
                    gaze_data_queue.append(gaze_data)
            
            # Draw gaze pointer
            dot_window.fill(0)
            
            if calibration_in_progress:
                # During calibration: show locked center with countdown
                remaining = calibration_duration - (current_time - calibration_start_time)
                cv2.circle(dot_window, (smooth_x, smooth_y), 25, (0, 255, 255), -1)  # Yellow during calibration
                cv2.circle(dot_window, (smooth_x, smooth_y), 27, (255, 255, 255), 3)
                
                # Draw countdown circle
                progress = (current_time - calibration_start_time) / calibration_duration
                angle = int(360 * progress)
                cv2.ellipse(dot_window, (smooth_x, smooth_y), (40, 40), 0, 0, angle, (0, 255, 0), 3)
            else:
                # Normal mode: red dot
                cv2.circle(dot_window, (smooth_x, smooth_y), 20, (0, 0, 255), -1)
                cv2.circle(dot_window, (smooth_x, smooth_y), 22, (255, 255, 255), 2)
            
            cv2.imshow('Gaze Pointer', dot_window)
            
            for idx in LEFT_IRIS + RIGHT_IRIS:
                x = int(landmarks[idx].x * frame_w)
                y = int(landmarks[idx].y * frame_h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            # Display information
            if calibration_in_progress:
                remaining = calibration_duration - (current_time - calibration_start_time)
                cv2.putText(frame, f"CALIBRATING - Look at CENTER", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Time remaining: {remaining:.1f}s", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Samples collected: {len(calibration_samples)}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                cv2.putText(frame, f"Gaze: ({smooth_x}, {smooth_y})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Normalized: ({normalized_x:.3f}, {normalized_y:.3f})", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Distance: {distance:.2f}m | Confidence: {confidence:.2f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Sensitivity: {sensitivity_multiplier:.1f}x", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                if not calibration_active:  # Only show offset after calibration
                    cv2.putText(frame, f"Calibration offset: X={gaze_vector_offset[0]:.4f}, Y={gaze_vector_offset[1]:.4f}", 
                               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
            
            if not calibration_in_progress:
                print(f"\rGaze: X={smooth_x:4d}, Y={smooth_y:4d} | Norm: ({normalized_x:.3f}, {normalized_y:.3f}) | Conf: {confidence:.2f} | Dist: {distance:.2f}m", 
                      end="", flush=True)
            else:
                remaining = calibration_duration - (current_time - calibration_start_time)
                print(f"\rCalibrating... {remaining:.1f}s remaining | Samples: {len(calibration_samples)}", 
                      end="", flush=True)
            
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
        # Recalibrate - reset and start new calibration
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
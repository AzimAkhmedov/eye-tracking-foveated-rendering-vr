import cv2
import mediapipe as mp
import numpy as np
import pyautogui

screen_w, screen_h = pyautogui.size()

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

def project_gaze_to_screen(gaze_vector, rotation_vec, distance, frame_w, frame_h):
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    
    gaze_world = rotation_mat @ gaze_vector
    
    monitor_distance = 0.6  
    scale_factor = distance / monitor_distance
    
    sensitivity_x = 10.5
    sensitivity_y = 12.5
    
    screen_x = screen_w / 2 + gaze_world[0] * screen_w * sensitivity_x * scale_factor
    screen_y = (screen_h / 2 + gaze_world[1] * screen_h * sensitivity_y * scale_factor)
    
    return int(screen_x), int(screen_y)

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
print("\nИнструкции:")
print("- Смотрите на камеру и двигайте взглядом")
print("- Красная точка показывает направление вашего взгляда")
print("- Нажмите 'q' для выхода")
print("- Нажмите 'c' для калибровки центра")
print("- Нажмите '+/-' для изменения чувствительности")
print("=" * 60)

offset_x, offset_y = 0, 0
sensitivity_multiplier = 1.0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Не удалось получить кадр с камеры")
        break
    
    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark
        
        try:
            gaze_vector, rotation_vec, distance = calculate_gaze_direction(landmarks, frame_w, frame_h)
            
            gaze_x, gaze_y = project_gaze_to_screen(gaze_vector, rotation_vec, distance, frame_w, frame_h)
            
            gaze_x += offset_x
            gaze_y += offset_y
            
            gaze_x = max(0, min(screen_w - 1, gaze_x))
            gaze_y = max(0, min(screen_h - 1, gaze_y))
            
            smooth_x = int(alpha * gaze_x + (1 - alpha) * smooth_x)
            smooth_y = int(alpha * gaze_y + (1 - alpha) * smooth_y)
            
            dot_window.fill(0)
            cv2.circle(dot_window, (smooth_x, smooth_y), 20, (0, 0, 255), -1)
            cv2.circle(dot_window, (smooth_x, smooth_y), 22, (255, 255, 255), 2)
            cv2.imshow('Gaze Pointer', dot_window)
            
            for idx in LEFT_IRIS + RIGHT_IRIS:
                x = int(landmarks[idx].x * frame_w)
                y = int(landmarks[idx].y * frame_h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            cv2.putText(frame, f"Gaze: ({smooth_x}, {smooth_y})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Distance: {distance:.2f}m", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Sensitivity: {sensitivity_multiplier:.1f}x", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            print(f"\rGaze: X={smooth_x:4d}, Y={smooth_y:4d} | Dist: {distance:.2f}m", 
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
        offset_x = (screen_w // 2) - smooth_x
        offset_y = (screen_h // 2) - smooth_y
        print("\n[✓] Калибровка выполнена - центр установлен")
    elif key == ord('+') or key == ord('='):
        sensitivity_multiplier += 0.1
        print(f"\n[↑] Чувствительность: {sensitivity_multiplier:.1f}x")
    elif key == ord('-') or key == ord('_'):
        sensitivity_multiplier = max(0.1, sensitivity_multiplier - 0.1)
        print(f"\n[↓] Чувствительность: {sensitivity_multiplier:.1f}x")

cap.release()
cv2.destroyAllWindows()
print("\n\nПрограмма завершена.")
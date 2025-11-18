"""
Head-Tracking Window Minimizer
Minimizes active window when head tilts down
Press 'q' to quit, 'p' to pause/resume, 'c' to calibrate
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque
import platform


if platform.system() == "Windows":
    import pygetwindow as gw
    import pyautogui
elif platform.system() == "Darwin":  # macOS
    from AppKit import NSWorkspace
    import subprocess
elif platform.system() == "Linux":
    import subprocess

class HeadTracker:
    def __init__(self):
 
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        

        self.PITCH_THRESHOLD = 25  
        self.TIME_THRESHOLD = 0.3   
        self.COOLDOWN_TIME = 2.0   
        self.SMOOTHING_WINDOW = 3   
        
 
        self.FILTER_MODE = "blacklist"  
        
        # WHITELIST: Only minimize these apps (case-insensitive partial match)
        self.WHITELIST = [
            "edge",
            "netflix",
        ]
        
        # BLACKLIST: Never minimize these apps (case-insensitive partial match)
        self.BLACKLIST = [
            "python",
            "terminal",
            "cmd",
            "powershell",
            "vscode",
            "Visual Studio Code",
            "pycharm",
            "notepad",
            "sublime",
            "zoom",
            "teams",
            "slack",
            "zen",
            "chrome",
        ]
        

        self.pitch_history = deque(maxlen=self.SMOOTHING_WINDOW)
        self.head_down_start = None
        self.last_minimize_time = 0
        self.is_paused = False
        self.calibration_offset = 0
        self.frame_skip = 2  
        self.frame_count = 0
    
        self.total_minimizes = 0
        self.fps_history = deque(maxlen=30)
        
    def get_head_pose(self, face_landmarks, img_w, img_h):
        """Calculate head pitch angle from face landmarks"""
   
        nose_tip = face_landmarks.landmark[1]
        chin = face_landmarks.landmark[152]
        forehead = face_landmarks.landmark[10]
        
    
        nose_2d = (nose_tip.x * img_w, nose_tip.y * img_h)
        chin_2d = (chin.x * img_w, chin.y * img_h)
        forehead_2d = (forehead.x * img_w, forehead.y * img_h)
        
     
        face_height = chin_2d[1] - forehead_2d[1]
        nose_offset = nose_2d[1] - forehead_2d[1]
        
     
        pitch = (nose_offset / face_height - 0.5) * 90
        
        return pitch
    
    def smooth_angle(self, angle):
        """Apply moving average smoothing"""
        self.pitch_history.append(angle)
        return np.mean(self.pitch_history)
    
    def should_minimize_window(self, window_title):
        """Check if window should be minimized based on filter mode"""
        if not window_title:
            return False
            
        window_title_lower = window_title.lower()
        
        if self.FILTER_MODE == "all":
            return True
        
        elif self.FILTER_MODE == "whitelist":
        
            return any(term.lower() in window_title_lower for term in self.WHITELIST)
        
        elif self.FILTER_MODE == "blacklist":
        
            is_blacklisted = any(term.lower() in window_title_lower for term in self.BLACKLIST)
            if is_blacklisted:
                print(f"  → Blocked by blacklist: {window_title}")
            return not is_blacklisted
        
        return False
    
    def minimize_active_window(self):
        """Minimize the currently active window (platform-specific)"""
        try:
            if platform.system() == "Windows":
                active_window = gw.getActiveWindow()
                if active_window:
                    window_title = active_window.title
                    print(f"  → Active window: {window_title}")
                    if self.should_minimize_window(window_title):
                        active_window.minimize()
                        print(f"  ✓ Minimized: {window_title}")
                        return True
                    else:
                        print(f"  → Skipped (filtered): {window_title}")
                        return False
                        
            elif platform.system() == "Darwin":  
            
                script_get_title = '''
                tell application "System Events"
                    set frontApp to name of first application process whose frontmost is true
                    return frontApp
                end tell
                '''
                result = subprocess.run(["osascript", "-e", script_get_title], 
                                      capture_output=True, text=True)
                window_title = result.stdout.strip()
                print(f"  → Active window: {window_title}")
                
                if self.should_minimize_window(window_title):
                    script_minimize = '''
                    tell application "System Events"
                        set frontApp to name of first application process whose frontmost is true
                        tell process frontApp
                            set value of attribute "AXMinimized" of window 1 to true
                        end tell
                    end tell
                    '''
                    subprocess.run(["osascript", "-e", script_minimize], capture_output=True)
                    print(f"  ✓ Minimized: {window_title}")
                    return True
                else:
                    print(f"  → Skipped (filtered): {window_title}")
                    return False
                    
            elif platform.system() == "Linux":
         
                result = subprocess.run(["xdotool", "getactivewindow", "getwindowname"],
                                      capture_output=True, text=True)
                window_title = result.stdout.strip()
                print(f"  → Active window: {window_title}")
                
                if self.should_minimize_window(window_title):
                    subprocess.run(["xdotool", "getactivewindow", "windowminimize"], 
                                 capture_output=True)
                    print(f"  ✓ Minimized: {window_title}")
                    return True
                else:
                    print(f"  → Skipped (filtered): {window_title}")
                    return False
                    
        except Exception as e:
            print(f"Error minimizing window: {e}")
            return False
        return False
    
    def calibrate(self, pitch):
        """Set current head position as neutral"""
        self.calibration_offset = pitch
        print(f"✓ Calibrated! Neutral position set to {pitch:.1f}° (offset: {self.calibration_offset:.1f}°)")
    
    def draw_ui(self, frame, pitch, status, fps):
        """Draw overlay UI on frame"""
        h, w = frame.shape[:2]
        
    
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
       
        color = (0, 255, 0) if status == "Ready" else (0, 165, 255) if status == "Paused" else (0, 0, 255)
        
       
        y_offset = 40
        cv2.putText(frame, f"Status: {status}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 35
        cv2.putText(frame, f"Head Angle: {pitch:.1f}deg", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
     
        direction = "DOWN" if pitch > 0 else "UP"
        direction_color = (0, 0, 255) if pitch > self.PITCH_THRESHOLD else (0, 255, 0)
        cv2.putText(frame, f"Direction: {direction}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, direction_color, 2)
        y_offset += 25
        
        cv2.putText(frame, f"Threshold: {self.PITCH_THRESHOLD}deg", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 25
        cv2.putText(frame, f"Minimized: {self.total_minimizes}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 25
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
 
        center_x, center_y = w - 100, 100
        radius = 60
        
       
        cv2.ellipse(frame, (center_x, center_y), (radius, radius), 
                   0, -90, 90, (100, 100, 100), 3)
        
    
        angle_normalized = max(-90, min(90, pitch))
        cv2.ellipse(frame, (center_x, center_y), (radius, radius),
                   0, -90, angle_normalized, color, 5)
        
     
        threshold_angle = self.PITCH_THRESHOLD
        threshold_x = int(center_x + radius * math.sin(math.radians(threshold_angle)))
        threshold_y = int(center_y - radius * math.cos(math.radians(threshold_angle)))
        cv2.line(frame, (center_x, center_y), (threshold_x, threshold_y), (0, 0, 255), 2)
        
   
        help_text = "Q:Quit | P:Pause | C:Calibrate | +/-:Sensitivity"
        cv2.putText(frame, help_text, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main tracking loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("=" * 50)
        print("Head-Tracking Window Minimizer Started")
        print("=" * 50)
        print(f"Filter Mode: {self.FILTER_MODE.upper()}")
        if self.FILTER_MODE == "whitelist":
            print(f"Will ONLY minimize: {', '.join(self.WHITELIST)}")
        elif self.FILTER_MODE == "blacklist":
            print(f"Will NOT minimize: {', '.join(self.BLACKLIST)}")
        else:
            print("Will minimize ALL windows")
        print("=" * 50)
        print("Controls:")
        print("  Q - Quit")
        print("  P - Pause/Resume")
        print("  C - Calibrate neutral position")
        print("  + - Increase sensitivity (lower threshold)")
        print("  - - Decrease sensitivity (higher threshold)")
        print("=" * 50)
        print("NOTE: Positive angles = looking DOWN")
        print("      Negative angles = looking UP")
        print("=" * 50)
        
        while cap.isOpened():
            start_time = time.time()
            success, frame = cap.read()
            if not success:
                continue
            
            frame = cv2.flip(frame, 1) 
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
           
            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
              
                pitch = self.pitch_history[-1] if self.pitch_history else 0
                status = "Paused" if self.is_paused else "Ready"
                fps = np.mean(self.fps_history) if self.fps_history else 0
                frame = self.draw_ui(frame, pitch, status, fps)
                cv2.imshow('Head Tracker', frame)
            
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    self.is_paused = not self.is_paused
                    print(f"{'⏸ Paused' if self.is_paused else '▶ Resumed'}")
                
                continue
            
        
            results = self.face_mesh.process(rgb_frame)
            current_time = time.time()
            status = "No Face Detected"
            pitch = 0
            
            if results.multi_face_landmarks and not self.is_paused:
                face_landmarks = results.multi_face_landmarks[0]
                
          
                raw_pitch = self.get_head_pose(face_landmarks, w, h)
                pitch = self.smooth_angle(raw_pitch + self.calibration_offset)
                
               
                if pitch > self.PITCH_THRESHOLD:
                    if self.head_down_start is None:
                        self.head_down_start = current_time
                        print(f"⬇ Head tilted down (angle: {pitch:.1f}°)")
                    
                    time_down = current_time - self.head_down_start
                    cooldown_elapsed = current_time - self.last_minimize_time
                    
                    if time_down >= self.TIME_THRESHOLD and cooldown_elapsed > self.COOLDOWN_TIME:
                        print(f"\n⏱ Trigger condition met after {time_down:.1f}s")
                        if self.minimize_active_window():
                            self.total_minimizes += 1
                            self.last_minimize_time = current_time
                            print(f"✓ Window minimized! (Total: {self.total_minimizes})\n")
                        self.head_down_start = None
                        status = "Window Minimized!"
                    else:
                        remaining = self.TIME_THRESHOLD - time_down
                        status = f"Head Down ({remaining:.1f}s left)"
                else:
                    if self.head_down_start is not None:
                        print(f"⬆ Head back up (angle: {pitch:.1f}°)")
                    self.head_down_start = None
                    status = "Ready"
                
          
                for landmark in [1, 152, 10]: 
                    lm = face_landmarks.landmark[landmark]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            elif self.is_paused:
                status = "Paused"
                pitch = self.pitch_history[-1] if self.pitch_history else 0
            
         
            fps = 1 / (time.time() - start_time) if time.time() - start_time > 0 else 0
            self.fps_history.append(fps)
            
          
            frame = self.draw_ui(frame, pitch, status, np.mean(self.fps_history))
            
      
            cv2.imshow('Head Tracker', frame)
           
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self.is_paused = not self.is_paused
                print(f"{'⏸ Paused' if self.is_paused else '▶ Resumed'}")
            elif key == ord('c'):
                if results.multi_face_landmarks:
                    raw_pitch = self.get_head_pose(results.multi_face_landmarks[0], w, h)
                  
                    self.calibrate(-raw_pitch)
            elif key == ord('+') or key == ord('='):
                self.PITCH_THRESHOLD += 2
                print(f"Sensitivity decreased (threshold: {self.PITCH_THRESHOLD}°)")
            elif key == ord('-') or key == ord('_'):
                self.PITCH_THRESHOLD -= 2
                print(f"Sensitivity increased (threshold: {self.PITCH_THRESHOLD}°)")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✓ Session complete! Total windows minimized: {self.total_minimizes}")

if __name__ == "__main__":
    try:
        tracker = HeadTracker()
        tracker.run()
    except KeyboardInterrupt:
        print("\n✓ Stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have installed required packages:")
        print("pip install opencv-python mediapipe numpy pygetwindow pyautogui")
import cv2
import torch
import numpy as np
from datetime import datetime
import time

class RealTimeTrashDetector:
    def __init__(self, model_path=None, camera_id=0, conf_threshold=0.5):
        """
        Initialize the trash detector
        
        Args:
            model_path: Path to custom YOLOv5 model (.pt file)
                       If None, uses a pre-trained model
            camera_id: Camera device ID (0 for default, 1 for external, or video file path)
            conf_threshold: Confidence threshold for detections (0-1)
        """
        self.camera_id = camera_id
        self.conf_threshold = conf_threshold
        
        # Load the model
        print("Loading YOLOv5 model...")
        if model_path:
            # Load custom trash detection model
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                       path=model_path, force_reload=False)
        else:
            # Load pre-trained YOLOv5 model (you can replace with trash-specific model)
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            
            # Map COCO classes to trash categories (example mapping)
            self.trash_classes = {
                39: 'plastic_bottle',  # bottle in COCO
                43: 'can',            # can -> reinterpreted
                67: 'cell_phone',     # electronic waste
                44: 'bottle',         # wine glass -> bottle
                46: 'banana',         # organic waste
            }
        
        # Set model parameters
        self.model.conf = conf_threshold  # confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold
        self.model.classes = None  # all classes (or specify [39, 43, 67] for specific classes)
        
        # Initialize camera
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.prev_time = 0
        
        # Define colors for different trash types
        self.colors = {
            'plastic': (0, 255, 0),       # Green
            'bottle': (0, 255, 255),      # Yellow
            'can': (255, 0, 0),           # Blue
            'paper': (255, 255, 0),       # Cyan
            'glass': (255, 0, 255),       # Magenta
            'organic': (0, 165, 255),     # Orange
            'electronic': (0, 0, 255),    # Red
            'default': (128, 128, 128)    # Gray
        }
        
        # Trash classification mapping (customize based on your model)
        self.trash_mapping = {
            'plastic': ['plastic', 'plastic_bag', 'wrapper'],
            'bottle': ['bottle', 'plastic_bottle', 'glass_bottle'],
            'can': ['can', 'tin_can', 'aluminum_can'],
            'paper': ['paper', 'cardboard', 'newspaper'],
            'glass': ['glass', 'glass_bottle', 'jar'],
            'organic': ['food', 'banana', 'apple', 'organic'],
            'electronic': ['cell_phone', 'battery', 'electronic']
        }
        
        print("Trash detector initialized!")
    
    def get_color_for_class(self, class_name):
        """Get color for specific trash class"""
        for trash_type, keywords in self.trash_mapping.items():
            if any(keyword in class_name.lower() for keyword in keywords):
                return self.colors.get(trash_type, self.colors['default'])
        return self.colors['default']
    
    def initialize_camera(self):
        """Initialize camera with optimal settings"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_id}")
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Height
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # FPS
        
        # Get actual camera properties
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
        return True
    
    def calculate_fps(self):
        """Calculate and display FPS"""
        current_time = time.time()
        if self.prev_time > 0:
            self.fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        return self.fps
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on the frame
        
        Args:
            frame: Input frame
            detections: Pandas DataFrame with detection results
        """
        for _, det in detections.iterrows():
            # Extract detection info
            x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
            confidence = det['confidence']
            class_name = str(det['name'])
            
            # Get color for this class
            color = self.get_color_for_class(class_name)
            
            # Draw bounding box
            box_thickness = max(1, int(min(x2-x1, y2-y1) * 0.005))  # Dynamic thickness
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            
            # Prepare label text
            label = f"{class_name}: {confidence:.2f}"
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Draw label background
            cv2.rectangle(frame, 
                         (x1, y1 - text_height - 10),
                         (x1 + text_width + 10, y1),
                         color, -1)  # -1 fills the rectangle
            
            # Draw label text
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add small corner indicators (optional visual enhancement)
            corner_length = 15
            cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, 3)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, 3)
            cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, 3)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, 3)
            cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, 3)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, 3)
            cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, 3)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, 3)
    
    def process_frame(self, frame):
        """Process a single frame and return detections"""
        # Perform inference
        results = self.model(frame)
        
        # Convert to pandas DataFrame
        detections = results.pandas().xyxy[0]
        
        # Filter by confidence threshold
        detections = detections[detections['confidence'] >= self.conf_threshold]
        
        return detections
    
    def display_info_panel(self, frame, detections, fps):
        """Display information panel on the frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Display information
        info_y = 40
        line_height = 25
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Detections count - FIXED THIS LINE
        cv2.putText(frame, f"Detections: {len(detections)}", (20, info_y + line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Confidence threshold
        cv2.putText(frame, f"Confidence: {self.conf_threshold:.2f}", (20, info_y + line_height*2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Current time
        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"Time: {current_time}", (20, info_y + line_height*3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        
        # Display detection classes (if any)
        if len(detections) > 0:
            unique_classes = detections['name'].unique()[:3]  # Show first 3 unique classes
            classes_text = ", ".join(unique_classes)
            if len(detections['name'].unique()) > 3:
                classes_text += "..."
            cv2.putText(frame, f"Classes: {classes_text}", (20, info_y + line_height*4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    
    def run(self):
        """Main loop for real-time detection"""
        if not self.initialize_camera():
            return
        
        print("\n" + "="*50)
        print("Trash Detection System Started!")
        print("Press 'q' to quit")
        print("Press '+' to increase confidence threshold")
        print("Press '-' to decrease confidence threshold")
        print("Press 's' to save current frame")
        print("Press 'c' to clear all detections")
        print("="*50 + "\n")
        
        # Create named window with properties
        cv2.namedWindow('Real-time Trash Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Real-time Trash Detection', 1280, 720)
        
        while True:
            # Read frame from camera
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Calculate FPS
            fps = self.calculate_fps()
            
            # Process frame (detect trash)
            detections = self.process_frame(frame)
            
            # Draw detections on frame
            self.draw_detections(frame, detections)
            
            # Display information panel
            self.display_info_panel(frame, detections, fps)
            
            # Show frame
            cv2.imshow('Real-time Trash Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                break
            elif key == ord('+'):  # Increase confidence threshold
                self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
                self.model.conf = self.conf_threshold
                print(f"Confidence threshold increased to: {self.conf_threshold:.2f}")
            elif key == ord('-'):  # Decrease confidence threshold
                self.conf_threshold = max(0.1, self.conf_threshold - 0.05)
                self.model.conf = self.conf_threshold
                print(f"Confidence threshold decreased to: {self.conf_threshold:.2f}")
            elif key == ord('s'):  # Save current frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"trash_detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as: {filename}")
            elif key == ord('c'):  # Clear/reset
                # In this case, just print message
                print("Cleared display (detections will refresh next frame)")
            
            self.frame_count += 1
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nTrash Detection System Stopped.")

# ==================== SIMPLIFIED VERSION ====================
# If you're having issues with the full version, try this minimal version first:

def simple_trash_detector():
    """Minimal working version - just 15 lines of core code"""
    import cv2
    import torch
    
    # Load model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects
        results = model(frame)
        
        # Draw results
        for *box, conf, cls in results.xyxy[0]:
            if conf > 0.5:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{model.names[int(cls)]}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show FPS
        cv2.putText(frame, "Trash Detector - Press 'q' to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow('Trash Detector', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Simple detector (recommended for testing)")
    print("2. Full featured detector")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        simple_trash_detector()
    elif choice == "2":
        detector = RealTimeTrashDetector()
        detector.run()
    else:
        print("Invalid choice. Running simple version...")
        simple_trash_detector()
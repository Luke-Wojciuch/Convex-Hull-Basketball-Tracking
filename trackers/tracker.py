from ultralytics import YOLO
import supervision as sv
import pickle, os, numpy as np
import cv2
from utils.bbox_utils import get_bbox_width, get_center_of_bbox
from sklearn.cluster import KMeans

class SmoothedObject:
    def __init__(self, initial_bbox, alpha=0.8):
        self.previous_bbox = np.array(initial_bbox)  # Ensure it's a NumPy array
        self.alpha = alpha  # Smoothing factor (0.0 to 1.0)

    def smooth_bbox(self, new_bbox):
        new_bbox = np.array(new_bbox)  # Convert new_bbox to NumPy array
        self.previous_bbox = self.alpha * self.previous_bbox + (1 - self.alpha) * new_bbox
        return self.previous_bbox.tolist()  # Convert back to a list


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path).to('cuda')
        self.tracker = sv.ByteTrack(lost_track_buffer= 60) # More stable tracking
        self.smoothed_objects = {}  # Store smoothed bboxes for each track_id
        self.last_seen_positions = {}  # Store last known positions
        self.team_colors = {}
        self.team_refs = []

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.15)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        tracks = {"players": [], "referees": []}


        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            detections_supervision = sv.Detections.from_ultralytics(detection)
            detection_with_track = self.tracker.update_with_detections(detections_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})

            # Process detections
            for frame_detection in detection_with_track:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['Player']:
                    if track_id not in self.smoothed_objects:
                        self.smoothed_objects[track_id] = SmoothedObject(bbox)

                    smoothed_bbox = self.smoothed_objects[track_id].smooth_bbox(bbox)
                    player_crop = self.extract_crop(frames[frame_num], smoothed_bbox)
                    dominant_color = self.get_dominant_color(player_crop)

                    if track_id not in self.team_colors:
                        if len(self.team_refs) < 2:
                            self.team_refs.append(dominant_color)
                            self.team_colors[track_id] = dominant_color
                        else:
                            dist0 = np.linalg.norm(np.array(dominant_color) - np.array(self.team_refs[0]))
                            dist1 = np.linalg.norm(np.array(dominant_color) - np.array(self.team_refs[1]))
                            self.team_colors[track_id] = self.team_refs[0] if dist0 < dist1 else self.team_refs[1]

                    tracks["players"][frame_num][track_id] = {
                        "bbox": smoothed_bbox,
                        "team_color": self.team_colors[track_id]
                    }

                elif cls_id == cls_names_inv['Ref']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def extract_crop(self, frame, bbox):
        """
        Extract a crop from a frame given a bounding box, focusing on the jersey area

        Args:
            frame: The full video frame
            bbox: Bounding box coordinates [x1, y1, x2, y2]

        Returns:
            The cropped image of the jersey area in RGB format
        """
        # Convert bbox coordinates to integers
        x1, y1, x2, y2 = map(int, bbox)

        # Get the dimensions of the bounding box
        box_width = x2 - x1
        box_height = y2 - y1

        # Define the jersey region (middle section of the player)
        # Avoid the head (top 30%) and legs (bottom 35%)
        y_top = int(y1 + 0.3 * box_height)  # Start below the head
        y_bottom = int(y1 + 0.65 * box_height)  # End above the legs

        # Narrow horizontally to focus on the jersey center (avoid arms)
        x_left = int(x1 + 0.3 * box_width)
        x_right = int(x1 + 0.7 * box_width)

        # Ensure coordinates are within frame boundaries
        h, w, _ = frame.shape
        y_top = max(0, min(y_top, h - 1))
        y_bottom = max(0, min(y_bottom, h - 1))
        x_left = max(0, min(x_left, w - 1))
        x_right = max(0, min(x_right, w - 1))

        # Check if crop dimensions are valid
        if y_bottom <= y_top or x_right <= x_left:
            # Return a small default crop if dimensions are invalid
            return np.zeros((10, 10, 3), dtype=np.uint8)

        # Extract the crop
        crop = frame[y_top:y_bottom, x_left:x_right]

        # Convert from BGR to RGB (OpenCV loads images as BGR)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        return crop_rgb

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = tuple(map(int, player["team_color"]))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            output_video_frames.append(frame)

        return output_video_frames

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame, (x_center, y2), (int(width), int(0.35 * width)), 0.0, -45, 235, color, 2, cv2.LINE_4)

        return frame


    def get_dominant_color(self, crop, k=2):
        """
        Extract dominant color from a player crop by focusing on the jersey area
        """
        # Get crop dimensions
        h, w, _ = crop.shape

        # Extract jersey area (middle section)
        jersey_crop = crop[int(0.3 * h):int(0.65 * h), int(0.3 * w):int(0.7 * w)]

        # Apply optional blur to reduce noise
        jersey_crop = cv2.GaussianBlur(jersey_crop, (5, 5), 0)

        # Reshape for KMeans
        pixels = jersey_crop.reshape((-1, 3))

        # Run KMeans
        kmeans = KMeans(n_clusters=k, random_state=42).fit(pixels)
        dominant = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]

        # Return as RGB tuple
        return tuple(map(int, dominant))
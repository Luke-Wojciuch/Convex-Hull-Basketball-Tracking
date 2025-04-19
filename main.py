import cv2
from utils.video_utils import read_video, save_video
from trackers.tracker import Tracker
from Camera_Movement import CameraMovementEstimator
from Perspective_Transform import CourtMapper, click_rectangle_template

def main():
    # Video Inputs
    input_video = "Rockets_Demo.mp4"


    # Find Which Side of the court and use the respective image
    chosen_side = ask_user_court_side_with_image()

    if(chosen_side == "right"):
        target_image_path = "Images/Half_Court_Right.jpg"
    else:
        target_image_path = "Images/Half_Court_Left.jpg"


    # Load court diagram
    temp_img = cv2.imread(target_image_path)

    # Read the video frames
    video_frames = read_video(input_video)
    print("Read Video Successfully!")

    # first frame for court corner selection
    frame = video_frames[0]

    # Click Corners of the Court
    print("Click the four corners of the court in the video frame in order:")
    print("1. Top-Left, 2. Top-Right, 3. Bottom-Right, 4. Bottom-Left")
    video_corners = click_rectangle_template(frame)

    # Draw Selected Court
    frame_with_polyline = video_frames[0].copy()
    cv2.polylines(frame_with_polyline, [video_corners.reshape((-1, 1, 2))],
                  True, (255, 255, 255), 2)
    cv2.imshow("Selected Court", frame_with_polyline)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Create CourtMapper
    court_mapper = CourtMapper(video_corners,frame,temp_img)
    court_mapper.show_warping()

    # Track objects
    tracker = Tracker("Model_Output/yolov8n_training11/weights/best.pt")
    tracks = tracker.get_object_tracks(video_frames)
    print("Tracking Objects Complete")

    # Handle camera movement discrepancies
    camera = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera.get_camera_movement(video_frames)
    camera.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    print("Camera Movement Adjustment Complete")

    # Create a list to store the radar view frames
    radar_frames = []
    player_tracks = tracks['players']
    for track in player_tracks:
        radar_frames.append(court_mapper.draw_players(track))


    # Save the radar view video
    output_video_path = "runs/detect/predict/radar_view.mp4"
    save_video(radar_frames, output_video_path)
    print(f"Radar view video saved to {output_video_path}")

    #Save the tracking video with annotations
    annotated_frames = tracker.draw_annotations(video_frames, tracks)
    output_video_path_tracking = "runs/detect/predict/tracking_view.mp4"
    save_video(annotated_frames, output_video_path_tracking)
    print(f"Tracking video saved to {output_video_path_tracking}")


def ask_user_court_side_with_image():
    court_img = cv2.imread("Images/Full_Court.jpg")
    height, width = court_img.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_color = (30, 30, 30)
    bg_color = (240, 240, 240)

    instruction_text = "Click on the half court you wish to track"
    (text_width, _), _ = cv2.getTextSize(instruction_text, font, font_scale, thickness)
    center_x = court_img.shape[1] // 2
    cv2.rectangle(court_img, (center_x - text_width // 2 - 10, 20), (center_x + text_width // 2 + 10, 60), bg_color, -1)
    cv2.putText(court_img, instruction_text, (center_x - text_width // 2, 50), font, font_scale, text_color, thickness,
                cv2.LINE_AA)

    # Left
    cv2.rectangle(court_img, (20, 190), (200, 235), bg_color, -1)
    cv2.putText(court_img, "Left Side", (30, 225), font, font_scale, (0, 120, 0), thickness, cv2.LINE_AA)

    # Right
    cv2.rectangle(court_img, (490, 190), (710, 235), bg_color, -1)
    cv2.putText(court_img, "Right Side", (500, 225), font, font_scale, (0, 0, 150), thickness, cv2.LINE_AA)

    selected = {}

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected['side'] = 'left' if x < width // 2 else 'right'
            cv2.destroyAllWindows()

    cv2.imshow("Which side is your video showing?", court_img)
    cv2.setMouseCallback("Which side is your video showing?", on_click)
    cv2.waitKey(0)

    return selected.get('side', 'left')


if __name__ == "__main__":
    main()

import numpy as np
import cv2


class CourtMapper:
    def __init__(self, video_corners, video_img, radar_img):
        self.source_vertices = np.array(video_corners, dtype=np.float32)

        rh, rw = radar_img.shape[:2]
        self.radar_corners = np.array([
            [0, 0],  # Top-left
            [rw, 0],  # Top-right
            [rw, rh],  # Bottom-right
            [0, rh]  # Bottom-left
        ], dtype=np.float32)

        self.H, self.status = cv2.findHomography(self.source_vertices, self.radar_corners)
        self.video_img = video_img
        self.radar_img = radar_img

    def show_warping(self):
        warped = cv2.warpPerspective(self.video_img, self.H, (self.radar_img.shape[1], self.radar_img.shape[0]))

        overlay = self.radar_img.copy()
        blend = cv2.addWeighted(overlay, 0.3, warped, 0.7, 0)

        cv2.imshow('Radar + Warped Court', blend)
        cv2.imwrite('runs/detect/predict/blended-court.png', blend)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def withinCourt(self,bbox):
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        # Check if the center point is inside the court
        court_polygon = np.array(self.source_vertices, dtype=np.int32)
        result = cv2.pointPolygonTest(court_polygon, (center_x, center_y), False)

        return result >= 0

    def draw_players(self, player_dict):
        player_radar = self.radar_img.copy()
        player_coords = []  # Collect all player coordinates
        player_colors = []  # Collect all player colors
        valid_players = []

        for player_id, player_data in player_dict.items():
            bbox = player_data['bbox']
            color = tuple(map(int, player_data["team_color"]))
            if self.withinCourt(bbox):
                x = (bbox[0] + bbox[2]) / 2
                y = (bbox[1] + bbox[3]) / 2
                player_coords.append([x, y])
                player_colors.append(color)
                valid_players.append(player_id)

        # Handle the case when player_coords is empty
        if not player_coords:
            return player_radar

        # Convert player_coords to a numpy array with shape (N, 2)
        player_coords_array = np.array(player_coords)

        # Transpose to match the expected shape (2, N) for apply_homography
        player_coords_array = player_coords_array.T

        assert player_coords_array.shape[0] == 2, "Coordinate array must have shape (2, N)"

        # Apply homography
        new_player_coords = apply_homography(self.H, player_coords_array)

        team_coords = {}
        for i in range(new_player_coords.shape[1]):
            coord = (int(new_player_coords[0, i]), int(new_player_coords[1, i]))
            color = player_colors[i]
            if color not in team_coords:
                team_coords[color] = []
            team_coords[color].append(coord)

        overlay = player_radar.copy()

        for color, coords in team_coords.items():
            if len(coords) >= 3:
                pts = np.array(coords, dtype=np.int32)
                hull = cv2.convexHull(pts)

                # Draw polygon on overlay
                hull_color = color[::-1]  # RGB to BGR
                cv2.fillConvexPoly(overlay, hull, hull_color)

                cv2.polylines(player_radar, [hull], isClosed=True, color=hull_color, thickness=2)

        # Blend overlay with original image
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, player_radar, 1 - alpha, 0, player_radar)

        # draw the players
        for i in range(new_player_coords.shape[1]):
            transformed_x = int(new_player_coords[0, i])
            transformed_y = int(new_player_coords[1, i])
            color = player_colors[i]

            circle_radius = 10
            outline_radius = circle_radius + 2
            cv2.circle(player_radar, (transformed_x, transformed_y), outline_radius, (0, 0, 0), -1)
            cv2.circle(player_radar, (transformed_x, transformed_y), circle_radius, color, -1)

        return player_radar




def click_rectangle_template(image):
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow("Click Court Corners (TL, TR, BR, BL)", image)

    print("Click the corners of the court (Top-Left, Top-Right, Bottom-Right, Bottom-Left)")
    temp = image.copy()
    cv2.imshow("Click Court Corners (TL, TR, BR, BL)", temp)
    cv2.setMouseCallback("Click Court Corners (TL, TR, BR, BL)", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 4:
        raise ValueError("You need to click exactly 4 points!")

    return np.array(points, dtype=np.int32)



# Adapted from:
# Gaurav A Mohan, "Developing a basketball minimap for player tracking using broadcast data and applied homography"
# Medium, July 2021. https://gauravamohan.medium.com/developing-a-basketball-minimap-for-player-tracking-using-broadcast-data-and-applied-homography-433183b9b995
def apply_homography(H, pts):
    assert (H.shape == (3, 3))
    assert (pts.shape[0] == 2)
    assert (pts.shape[1] >= 1)

    # Convert to homogeneous coordinates for cleaner computation
    homogeneous_pts = np.vstack((pts, np.ones(pts.shape[1])))

    # Apply homography matrix multiplication
    transformed_homogeneous = H @ homogeneous_pts

    # Normalize by dividing by the third coordinate
    w = transformed_homogeneous[2, :]
    # Handle potential division by zero with a small epsilon
    w[np.abs(w) < 1e-10] = 1e-10

    # Convert back from homogeneous coordinates
    tpts = transformed_homogeneous[:2, :] / w

    # make sure transformed pts are correct dimension
    assert (tpts.shape[0] == 2)
    assert (tpts.shape[1] == pts.shape[1])

    return tpts

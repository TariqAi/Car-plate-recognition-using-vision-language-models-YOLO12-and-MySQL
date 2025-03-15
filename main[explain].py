import cv2  # OpenCV for video and image processing
import os  # For operating system-level operations like directory creation
import base64  # To encode image data into base64 for API requests
import threading  # For running concurrent tasks (like analyzing data and processing frames)
import mysql.connector  # To connect to and interact with a MySQL database
from time import time  # For tracking time intervals, used in speed calculations
import numpy as np  # For handling numerical operations, specifically for speed calculation
from ultralytics.solutions.solutions import BaseSolution  # Import the base class for vehicle detection
from ultralytics.utils.plotting import Annotator, colors  # Used for drawing bounding boxes and annotations
from datetime import datetime  # For working with date and time (storing in the database)
from shapely.geometry import LineString  # For geometric operations (detecting if vehicles cross regions)
from openai import OpenAI  # For interacting with OpenAI's API to extract vehicle details

# Set up OpenAI API Key (securely)
OPENROUTER_API_KEY = ""  # Replace with your actual API key
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY  # Set the API key as an environment variable

# MySQL Database Configuration (parameters to connect to the database)
DB_CONFIG = {
    "host": "localhost",
    "user": "root",  # MySQL username
    "password": "",  # MySQL password (empty here, change as per your configuration)
    "database": "vehicle_data"  # Name of the database where vehicle data will be stored
}

# Function to initialize the MySQL database and table
def initialize_database():
    """Creates the database and table if they do not exist."""
    try:
        # Connect to the MySQL server (without specifying the database)
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        )
        cursor = conn.cursor()

        # Create the database if it does not exist
        cursor.execute("CREATE DATABASE IF NOT EXISTS vehicle_data")
        cursor.close()
        conn.close()

        # Connect to the database now
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Create the table to store vehicle records if it does not exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_records (
            id INT AUTO_INCREMENT PRIMARY KEY,
            track_id INT,
            speed INT,
            date_time DATETIME,
            vehicle_model VARCHAR(255),
            vehicle_color VARCHAR(100),
            vehicle_company VARCHAR(255),
            number_plate VARCHAR(50)
        )
        """)
        conn.commit()  # Save the changes
        cursor.close()
        conn.close()  # Close the database connection
        print("✅ Database and table are ready!")  # Success message

    except Exception as e:
        print(f"❌ Database Initialization Error: {e}")  # Error message if something goes wrong

# Call the function to initialize the database & table
initialize_database()

# Function to insert detected vehicle data into the database
def insert_into_database(track_id, speed, timestamp, model, color, company, number_plate):
    """Insert detected vehicle details into MySQL database."""
    try:
        # Connect to the MySQL database
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # SQL query to insert data
        query = """
        INSERT INTO vehicle_records (track_id, speed, date_time, vehicle_model, vehicle_color, vehicle_company, number_plate)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        values = (track_id, speed, timestamp, model, color, company, number_plate)

        # Execute the query
        cursor.execute(query, values)
        conn.commit()  # Save the changes
        cursor.close()
        conn.close()  # Close the database connection
        print(f"✅ Data inserted for Track ID: {track_id}")  # Success message
    except Exception as e:
        print(f"❌ Database Insert Error: {e}")  # Error message if something goes wrong

# Function to fetch the license plate of a vehicle using the track_id
def get_license_plate(track_id):
    """Fetch the license plate number for a given track_id from the database."""
    try:
        # Connect to the MySQL database
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # SQL query to retrieve the license plate number of a vehicle
        query = "SELECT number_plate FROM vehicle_records WHERE track_id = %s ORDER BY date_time DESC LIMIT 1"
        cursor.execute(query, (track_id,))  # Execute query with track_id as parameter
        result = cursor.fetchone()  # Fetch the result

        cursor.close()
        conn.close()  # Close the connection

        if result and result[0]:  # If a result is found, return the license plate
            return result[0]
        return "Unknown"  # Return "Unknown" if no result is found
    except Exception as e:
        print(f"❌ Error fetching license plate for Track ID {track_id}: {e}")
        return "Unknown"  # Return "Unknown" in case of an error

# Class to estimate speed and handle vehicle detection (inherits from BaseSolution)
class SpeedEstimator(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Initialize parent class (BaseSolution)
        self.initialize_region()  # Define the region where vehicles are monitored (custom region)
        self.spd = {}  # Dictionary to store the speed of each vehicle
        self.trkd_ids = []  # List to store IDs of vehicles currently being tracked
        self.trk_pt = {}  # Dictionary to store timestamps of vehicles
        self.trk_pp = {}  # Dictionary to store previous positions of vehicles
        self.saved_ids = set()  # Set to track saved vehicle images
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,  # OpenAI API client setup with API key
        )

        # Ensure that the directory for saving cropped images exists
        os.makedirs("crop", exist_ok=True)

    # Function to analyze images and extract vehicle information using OpenAI API
    def analyze_and_save_response(self, image_path, track_id, speed, timestamp):
        """Analyzes the image with OpenAI API and saves response to MySQL."""
        try:
            # Open the image and convert it to base64
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            # Send a request to OpenAI to extract vehicle information from the image
            completion = self.client.chat.completions.create(
                model="google/gemini-2.0-pro-exp-02-05:free",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract ONLY these details:\n| Vehicle Model | Color | Company | Number Plate |\n|--------------|--------|---------|--------------|"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, "description": "Detected vehicle"}
                    ]
                }]
            )

            # Process the API response
            response_text = completion.choices[0].message.content.strip()

            # Extract relevant data from the response
            valid_rows = [
                row.split("|")[1:-1] for row in response_text.split("\n")
                if "|" in row and "Vehicle Model" not in row and "---" not in row
            ]
            vehicle_info = valid_rows[0] if valid_rows else ["Unknown", "Unknown", "Unknown", "Unknown"]

            # Insert extracted data into the database
            insert_into_database(track_id, speed, timestamp, vehicle_info[0], vehicle_info[1], vehicle_info[2], vehicle_info[3])

        except Exception as e:
            print(f"❌ Error invoking OpenAI API: {e}")

    # Function to estimate speed based on the position of vehicles in the video frame
    def estimate_speed(self, im0):
        """Estimate speed of detected vehicles in the video frame."""
        self.annotator = Annotator(im0, line_width=self.line_width)  # Annotator for drawing bounding boxes and text
        self.extract_tracks(im0)  # Extract vehicle tracks (bounding boxes)
        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get the current timestamp

        # Loop through the detected vehicles and their bounding boxes
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)  # Store vehicle tracking data

            if track_id not in self.trk_pt:  # Initialize timestamp for new vehicles
                self.trk_pt[track_id] = time()
            if track_id not in self.trk_pp:  # Initialize previous position for new vehicles
                self.trk_pp[track_id] = box  

            prev_pos = self.trk_pp[track_id]  # Get previous position
            curr_pos = box  # Get current position

            # Check if the vehicle has crossed the region of interest
            if LineString([prev_pos[:2], curr_pos[:2]]).intersects(LineString(self.region)):
                direction = "known"
            else:
                direction = "unknown"

            # If the vehicle is moving in the known direction, estimate its speed
            if direction == "known" and track_id not in self.trkd_ids:
                self.trkd_ids.append(track_id)
                time_difference = time() - self.trk_pt[track_id]
                if time_difference > 0:
                    speed = np.linalg.norm(np.array(curr_pos[:2]) - np.array(prev_pos[:2])) / time_difference
                    self.spd[track_id] = round(speed)  # Store the speed value

            self.trk_pt[track_id] = time()  # Update the timestamp for the current frame
            self.trk_pp[track_id] = curr_pos  # Update the previous position

            speed_value = self.spd.get(track_id, 0)  # Get the speed value for the vehicle
            label = f"ID: {track_id} {speed_value} km/h"  # Annotate the speed on the frame
            self.annotator.box_label(box, label=label, color=colors(track_id, True))  # Draw bounding box and speed label

            # Fetch the license plate number for the vehicle
            license_plate = get_license_plate(track_id)

            # Calculate the center of the bounding box to display the license plate
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Display the license plate text on the image
            text_size, _ = cv2.getTextSize(license_plate, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_width, text_height = text_size
            text_x = center_x - text_width // 2
            text_y = center_y + text_height // 2

            # Draw a background rectangle for better visibility of the license plate text
            cv2.rectangle(
                im0,
                (text_x - 5, text_y - text_height - 5),
                (text_x + text_width + 5, text_y + 5),
                (0, 0, 0),  # Black background
                -1  # Filled rectangle
            )

            # Put the license plate text on the image
            cv2.putText(
                im0,
                license_plate,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # Font scale
                (255, 255, 255),  # White text
                2  # Thickness of the text
            )

            # If the vehicle data is not already saved, save the cropped image
            if track_id in self.spd and track_id not in self.saved_ids:
                x1, y1, x2, y2 = map(int, box)
                cropped_image = im0[y1:y2, x1:x2]  # Crop the vehicle's image from the frame

                # If the cropped image has data, save it as a file
                if cropped_image.size != 0:
                    image_filename = f"crop/{track_id}_{speed_value}kmh.jpg"
                    cv2.imwrite(image_filename, cropped_image)  # Save the cropped image
                    print(f"📷 Saved image: {image_filename}")

                    # Analyze the image and save the response in a new thread (non-blocking)
                    threading.Thread(
                        target=self.analyze_and_save_response,
                        args=(image_filename, track_id, speed_value, current_time),
                        daemon=True  # Run this thread in the background
                    ).start()

                    self.saved_ids.add(track_id)  # Mark the vehicle ID as saved

        self.display_output(im0)  # Display the annotated output on the screen
        return im0  # Return the annotated image

# Main execution block for video processing
cap = cv2.VideoCapture('tc.mp4')  # Open the video file

# Define the region of interest (ROI) for vehicle detection
region_points = [(0, 119), (1018, 119)]

# Initialize the SpeedEstimator class with the defined region and YOLO model
speed_obj = SpeedEstimator(region=region_points, model="yolo12s.pt", line_width=2)

# Get video properties (frame width, height, and frames per second)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the output video file and its codec
output_video_file = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 video format
out = cv2.VideoWriter(output_video_file, fourcc, fps, (1020, 500))  # Initialize the video writer

while True:
    ret, frame = cap.read()  # Read the next frame from the video
    if not ret:
        break  # Exit if the frame cannot be read

    frame = cv2.resize(frame, (1020, 500))  # Resize the frame for consistency

    # Estimate the speed of vehicles in the current frame
    result = speed_obj.estimate_speed(frame)

    # Write the processed frame to the output video
    out.write(result)

    # Display the processed frame in a window
    cv2.imshow("Speed Estimation", result)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources after video processing
cap.release()
out.release()  # Release the VideoWriter to save the output video
cv2.destroyAllWindows()
print(f"✅ Video saved as {output_video_file}")

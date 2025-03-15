import mysql.connector  
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def initialize_database():
    """Creates the database and table if they do not exist."""
    try:
        conn = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME')
        )
        cursor = conn.cursor()

        # Create table
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

        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Database and table are ready!")

    except mysql.connector.Error as e:
        print(f"❌ Database Initialization Error: {e}")

def insert_into_database(track_id, speed, timestamp, model, color, company, number_plate):
    """Insert detected vehicle details into MySQL database."""
    try:
        conn = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME')
        )
        cursor = conn.cursor()

        query = """
        INSERT INTO vehicle_records (track_id, speed, date_time, vehicle_model, vehicle_color, vehicle_company, number_plate)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        values = (track_id, speed, timestamp, model, color, company, number_plate)

        cursor.execute(query, values)
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ Data inserted for Track ID: {track_id}")
    except mysql.connector.Error as e:
        print(f"❌ Database Insert Error: {e}")

def get_license_plate(track_id):
    """Fetch the license plate number for a given track_id from the database."""
    try:
        conn = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME')
        )
        cursor = conn.cursor()

        query = "SELECT number_plate FROM vehicle_records WHERE track_id = %s ORDER BY date_time DESC LIMIT 1"
        cursor.execute(query, (track_id,))
        result = cursor.fetchone()  

        cursor.close()
        conn.close()

        if result and result[0]:
            return result[0]
        return "Unknown"
    except mysql.connector.Error as e:
        print(f"❌ Error fetching license plate for Track ID {track_id}: {e}")
        return "Unknown"
    
    
def test_mysql_connection():
    """Test if the connection to MySQL is successful."""
    try:
        conn = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME')
        )
        if conn.is_connected():
            print("✅ Successfully connected to MySQL!")
        conn.close()
    except mysql.connector.Error as e:
        print(f"❌ MySQL Connection Error: {e}")


# Run the test function
test_mysql_connection()
initialize_database()
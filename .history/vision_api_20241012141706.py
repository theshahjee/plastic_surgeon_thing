import os
import json
from google.cloud import vision
from google.oauth2 import service_account

# Set the path to your Google Cloud service account key
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './path_to_your_service_account_key.json'

# Initialize the Vision API client
client = vision.ImageAnnotatorClient()

def detect_face_landmarks(image_path):
    """Detect face landmarks using Google Cloud Vision API."""
    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    # Create an image object for the Google Vision API
    image = vision.Image(content=content)

    # Perform face detection
    response = client.face_detection(image=image)
    faces = response.face_annotations

    landmarks = []
    for face in faces:
        for landmark in face.landmarks:
            landmarks.append({
                "type": landmark.type_.name,
                "position": {"x": landmark.position.x, "y": landmark.position.y}
            })

    if response.error.message:
        raise Exception(f"API error: {response.error.message}")

    return landmarks

# Example usage
image_path = './sample_image.jpg'
face_landmarks = detect_face_landmarks(image_path)

# Print detected landmarks
print(json.dumps(face_landmarks, indent=4))

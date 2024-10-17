import os
import json
from google.cloud import vision
from google.oauth2 import service_account

# Using ./ to refer to the credentials file in the current directory
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'./prime-depot-438408-s9-2958f9e45383.json'

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
image_path = './sample_image.jpg'  # Ensure this path is correct too
face_landmarks = detect_face_landmarks(image_path)

# Print detected landmarks
print(json.dumps(face_landmarks, indent=4))

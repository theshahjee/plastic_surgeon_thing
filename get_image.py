import face_recognition

# Load the image using face_recognition
image_path = "./sample_image.jpg"
image = face_recognition.load_image_file(image_path)

# Detect face locations
face_locations = face_recognition.face_locations(image)

if len(face_locations) == 0:
    print("No face detected in the image.")
else:
    # Loop over all face locations
    for face_location in face_locations:
        top, right, bottom, left = face_location

        # Convert to x, y, width, and height
        x = left
        y = top
        face_width = right - left
        face_height = bottom - top

        # Print the original bounding box values
        print(f"Bounding Box - Top: {top}, Right: {right}, Bottom: {bottom}, Left: {left}")

        # Print the face coordinates and dimensions
        print(f"Face found at X: {x}, Y: {y}, Width: {face_width}, Height: {face_height}")

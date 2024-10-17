import face_recognition
import base64
import json
import re
from PIL import Image, ImageDraw, ImageFont
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# Function to encode the image in base64
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None

# Function to extract JSON from the string returned by OpenAI
def extract_json_from_string(response_content):
    json_match = re.search(r'\[.*\]', response_content, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
    else:
        print("No valid JSON found in the response.")
        return None

# Function to format landmarks as string
def format_landmarks(landmarks):
    formatted_landmarks = []
    for region, points in landmarks.items():
        formatted_landmarks.append(f"{region}: {points}")
    return ", ".join(formatted_landmarks)

# Function to draw points on the image
def draw_points_on_image(image_path, points_data, output_path):
    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Try loading a font (you can change the path to a TTF file)
    try:
        font = ImageFont.truetype("arial.ttf", size=15)  # Adjust the font size as needed
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font

    # Loop through the points and mark them on the image
    for i, point in enumerate(points_data, 1):
        coordinates = point["coordinates"]
        x, y = coordinates

        # Draw a small dot (circle)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="red", outline="red")
        
        # Label the point with the number
        draw.text((x + 5, y - 5), f"{i}", fill="white", font=font)
    
    # Save the annotated image
    image.save(output_path)
    print(f"Annotated image saved at: {output_path}")
    return image

# Initialize OpenAI
openai_api_key = 'sk-gIMNPrI_kzWMwePo7eIK33WyLcb9YwRAF-8OLupYXoT3BlbkFJLOgRISb6xNI8VN1FGbqHcqy39B6aslj-5VWere7zoA'
chain = ChatOpenAI(api_key=openai_api_key, model="gpt-4o")



# Load image using face_recognition
image_path = "./sample_image.jpg"  # Path to your image
image = face_recognition.load_image_file(image_path)

# Get image dimensions (height, width)
image_height, image_width, _ = image.shape

# Detect face locations
face_locations = face_recognition.face_locations(image)

# Ensure a face is detected
if len(face_locations) > 0:
    # Get the first face's bounding box (top, right, bottom, left)
    top, right, bottom, left = face_locations[0]
    
    # Calculate face width and height
    face_width = right - left
    face_height = bottom - top
    
    # Format landmarks for the first face
    face_landmarks = face_recognition.face_landmarks(image)
    formatted_landmarks = format_landmarks(face_landmarks[0])
    
    # Encode the image in base64
    image_data = encode_image(image_path)
    
    if image_data:
        # Create the prompt template with the correct image and face dimensions
        prompt_template = PromptTemplate(
            input_variables=["image_data", "image_width", "image_height", "face_width", "face_height", "landmarks"],
            template="""
            Analyze the provided facial image and suggest areas where fat injections can be placed to create a more youthful appearance.
            The image size is {image_width}x{image_height} pixels, and the face region has a width of {face_width} pixels and height of {face_height} pixels.
            
            For each area, provide the following details:
            
            1. **Name of the region** (e.g., cheeks, under-eye, jawline).
            2. **Pixel coordinates (x, y)** of the site for fat injections, calculated relative to the face dimensions.
            3. The associated **meridian** that controls that region based on Traditional Chinese Medicine (TCM).
            4. A specific **acupuncture point** from TCM corresponding to that area.
            5. A **reason** explaining why this particular area benefits from fat injections.
            
            Please provide a minimum of 15 points for fat injection sites. Ensure the response is in valid JSON format.
            
            Image Data: {image_data}
            Face Landmarks: {landmarks}
            """
        )

        # Format the prompt with the image and face data
        prompt = prompt_template.format(
            image_data=image_data,
            image_width=image_width,
            image_height=image_height,
            face_width=face_width,
            face_height=face_height,
            landmarks=formatted_landmarks
        )

        # Invoke the model with the prompt
        response = chain([HumanMessage(content=prompt)])
        
        # Get the response content
        response_content = response.content
        print(response_content)

        # Extract and process the JSON data from the response
        fat_injection_data = extract_json_from_string(response_content)
        
        if fat_injection_data:
            # Annotate the image with points
            output_path = "./annotated_image.jpg"
            image_with_points = draw_points_on_image(image_path, fat_injection_data, output_path)
            image_with_points.show()

            # Print the extracted points and details
            for point in fat_injection_data:
                region = point["region"]
                coordinates = point["coordinates"]
                meridian = point["meridian"]
                acupuncture_point = point["acupuncture_point"]
                reason = point["reason"]
                print(f"Region: {region}, Coordinates: {coordinates}, Meridian: {meridian}, Acupuncture Point: {acupuncture_point}, Reason: {reason}")
else:
    print("No face detected in the image.")

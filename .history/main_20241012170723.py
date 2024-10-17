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
            You are an expert in facial aesthetics and Traditional Chinese Medicine (TCM). 
            Analyze the provided facial image with the following dimensions:

            - **Image width**: {image_width} pixels
            - **Image height**: {image_height} pixels
            - **Face width**: {face_width} pixels
            - **Face height**: {face_height} pixels.

            Your objective is to identify specific areas on the face where fat injections can enhance youthfulness and overall facial harmony. 

            For each suggested injection site, please provide detailed information in the following format:

            1. **Region Name**: The name of the facial area (e.g., cheeks, under-eye, jawline).
            
            2. **Pixel Coordinates (x, y)**: Precise coordinates relative to the entire image size (i.e., {image_width} x {image_height}). Ensure that these coordinates correspond to the specified region within the facial area.

            3. **Meridian Association**: Indicate the TCM meridian that corresponds to this facial region (e.g., Gall Bladder (GB), Stomach (St), Bladder (B)).

            4. **Acupuncture Point**: Provide a relevant acupuncture point from TCM that relates to this region (e.g., *Facial Beauty (St3)* or *Drilling Bamboo (B2)*).

            5. **Reason for Fat Injection**: Explain why fat injections are beneficial for this area (e.g., restoring volume, enhancing contour, reducing signs of aging, improving skin texture).

            **Instructions**:
            - Please list **at least 15 recommended injection sites** but feel free to suggest more if relevant.
            - Ensure that the response is formatted in valid JSON to facilitate easy parsing.
            - Include details from the facial landmarks provided below to inform your decisions.

            **Facial Landmarks**: {landmarks}

            **Example JSON Format**:
            [
                {{
                    "region": "Forehead",
                    "coordinates": [150, 100],
                    "meridian": "Gall Bladder (GB)",
                    "acupuncture_point": "GB14",
                    "reason": "Restoring volume to smooth out forehead lines and enhance youthful contour."
                }}
            ]

            **Image Data**: (for reference) {image_data}
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

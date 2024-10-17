import cv2
import base64
import json
import re
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None

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
    
def mark_coordinates_on_image(image_path, coordinates_list):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load the image {image_path}.")
        return

    for index, point in enumerate(coordinates_list):
        coordinates = point["coordinates"]
        x, y = coordinates[0], coordinates[1]
        radius = 10  
        cv2.circle(image, (x, y), radius, color=(0, 0, 255), thickness=-1)  
        font_scale = 0.6
        thickness = 2
        text_position = (x - 5, y + 5)
        cv2.putText(image, str(index + 1), text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.imshow("Image with Marked Points", image)
    output_path = "output_image_with_points.jpg"
    cv2.imwrite(output_path, image)
    print(f"Marked image saved as {output_path}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_face(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("No face detected.")
        return None
    
    for (x, y, w, h) in faces:
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        width, height = w, h
        return {"top_left": top_left, "bottom_right": bottom_right, "width": width, "height": height}

    return None

# Initialize OpenAI
openai_api_key = 'sk-gIMNPrI_kzWMwePo7eIK33WyLcb9YwRAF-8OLupYXoT3BlbkFJLOgRISb6xNI8VN1FGbqHcqy39B6aslj-5VWere7zoA'
chain = ChatOpenAI(api_key=openai_api_key, model="gpt-4o")

image_path = "./sample_image.jpg"
image_data = encode_image(image_path)

if image_data: 
    # Detect face
    face_info = detect_face(image_path)

    if face_info:
        top_left = face_info['top_left']
        bottom_right = face_info['bottom_right']
        width = face_info['width']
        height = face_info['height']

        # Create the prompt including the face information
        prompt_template = PromptTemplate(
            input_variables=["image_data", "top_left", "bottom_right", "width", "height"],
            template="""
            The image provided contains a face detected with the following dimensions:

            - Top-left coordinate: {top_left}
            - Bottom-right coordinate: {bottom_right}
            - Face Width: {width}
            - Face Height: {height}

            Please analyze this facial image and suggest areas where fat injections can be placed to create a more youthful appearance. For each area, provide the following details:

            1. **Name of the region** (e.g., cheeks, under-eye, jawline).
              
            2. Approximate **pixel coordinates (x, y)** of the site.
              
            3. The associated **meridian** that controls that region based on Traditional Chinese Medicine (TCM) (e.g., Gall Bladder (GB), Stomach (St), Bladder (B)).
              
            4. A specific **acupuncture point** from TCM corresponding to that area (e.g., *Facial Beauty (St3)* or *Drilling Bamboo (B2)*).
              
            5. A **reason** explaining why this particular area benefits from fat injections (e.g., restoring volume, enhancing contour, or reducing signs of aging).

            Provide at least 15 points for fat injection sites, in valid JSON format.

            Image Data: {image_data}
            """
        )

        prompt = prompt_template.format(
            image_data=image_data,
            top_left=top_left,
            bottom_right=bottom_right,
            width=width,
            height=height
        )

        response = chain([HumanMessage(content=prompt)])
        response_content = response.content

        fat_injection_data = extract_json_from_string(response_content)

        if fat_injection_data:
            mark_coordinates_on_image(image_path, fat_injection_data)
            for point in fat_injection_data:
                region = point["region"]
                coordinates = point["coordinates"]
                meridian = point["meridian"]
                acupuncture_point = point["acupuncture_point"]
                reason = point["reason"]
                
                print(f"Region: {region}, Coordinates: {coordinates}, Meridian: {meridian}, Acupuncture Point: {acupuncture_point}, Reason: {reason}")
    else:
        print("No face detected. Exiting program.")
else:
    print("Image data not provided. Exiting program.")

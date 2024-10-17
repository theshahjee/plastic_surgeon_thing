import base64
import json
import re
import cv2  # Import OpenCV
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# Function to encode the image to base64
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None

def extract_json_from_string(response_content):
    # Use regular expression to extract the JSON portion from the string
    json_match = re.search(r'\[.*\]', response_content, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(0)  # Get the matched JSON string
        try:
            # Parse the extracted JSON string into a Python object
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
    else:
        print("No valid JSON found in the response.")
        return None
    
# Initialize OpenAI
openai_api_key = 'sk-gIMNPrI_kzWMwePo7eIK33WyLcb9YwRAF-8OLupYXoT3BlbkFJLOgRISb6xNI8VN1FGbqHcqy39B6aslj-5VWere7zoA'

chain = ChatOpenAI(api_key=openai_api_key, model="gpt-4o")

# Encode the image
image_data = encode_image("./sample_image.jpg")

if image_data: 
    # Correct the template with escaped braces for JSON and prevent key errors
    prompt_template = PromptTemplate(
        input_variables=["image_data"],
        template="""
        Analyze the provided facial image and suggest areas where fat injections can be placed to create a more youthful appearance. For each area, provide the following details:

        1. **Name of the region** (e.g., cheeks, under-eye, jawline).
          
        2. Approximate **pixel coordinates (x, y)** of the site.
          
        3. The associated **meridian** that controls that region based on Traditional Chinese Medicine (TCM) (e.g., Gall Bladder (GB), Stomach (St), Bladder (B)).
          
        4. A specific **acupuncture point** from TCM corresponding to that area (e.g., *Facial Beauty (St3)* or *Drilling Bamboo (B2)*).
          
        5. A **reason** explaining why this particular area benefits from fat injections (e.g., restoring volume, enhancing contour, or reducing signs of aging).

        Please provide **a minimum of 15 points** for fat injection sites, but feel free to include more if applicable. 
        Ensure that the response is in valid JSON format.

        Image Data: {image_data}

        Sample format:
        [
            {{
                "region": "Forehead",
                "coordinates": [150, 100],
                "meridian": "Gall Bladder (GB)",
                "acupuncture_point": "GB14",
                "reason": "Restoring volume to smooth out forehead lines and enhance a youthful contour."
            }}
        ]
        """
    )

    # Format the prompt with the image data
    prompt = prompt_template.format(image_data=image_data)

    # Invoke the model with the prompt
    response = chain([HumanMessage(content=prompt)])

    # Get the response content
    response_content = response.content
    print(response_content)

    fat_injection_data = extract_json_from_string(response_content)

    if fat_injection_data:
        for point in fat_injection_data:
            region = point["region"]
            coordinates = point["coordinates"]
            meridian = point["meridian"]
            acupuncture_point = point["acupuncture_point"]
            reason = point["reason"]
            
            print(f"Region: {region}, Coordinates: {coordinates}, Meridian: {meridian}, Acupuncture Point: {acupuncture_point}, Reason: {reason}")
else:
    print("Image data not provided. Exiting program.")
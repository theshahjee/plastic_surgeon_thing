from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
import face_recognition
import base64
import json
import re
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
import io

app = FastAPI()

# Function to encode image in base64
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

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

# Function to format landmarks as a string
def format_landmarks(landmarks):
    formatted_landmarks = []
    for region, points in landmarks.items():
        formatted_landmarks.append(f"{region}: {points}")
    return ", ".join(formatted_landmarks)

# Function to draw points on the image
def draw_points_on_image(image, points_data):
    draw = ImageDraw.Draw(image)
    
    # Load font (adjust path if needed)
    try:
        font = ImageFont.truetype("arial.ttf", size=15)
    except IOError:
        font = ImageFont.load_default()

    for i, point in enumerate(points_data, 1):
        coordinates = point["coordinates"]
        x, y = coordinates

        # Draw circle on the point
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="red", outline="red")

        # Label the point with the number
        draw.text((x + 5, y - 5), f"{i}", fill="white", font=font)
    
    return image

# Initialize OpenAI
openai_api_key = 'YOUR_OPENAI_API_KEY'
chain = ChatOpenAI(api_key=openai_api_key, model="gpt-4")

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Load image
        image = face_recognition.load_image_file(file.file)
        pil_image = Image.fromarray(image)

        # Get image dimensions (height and width)
        image_height, image_width, _ = image.shape

        # Detect face locations and landmarks
        face_locations = face_recognition.face_locations(image)
        face_landmarks = face_recognition.face_landmarks(image)

        if len(face_locations) == 0:
            return JSONResponse(content={"error": "No face detected in the image."}, status_code=400)

        # Get the first face's bounding box
        top, right, bottom, left = face_locations[0]
        face_width = right - left
        face_height = bottom - top

        # Format landmarks for prompt
        formatted_landmarks = format_landmarks(face_landmarks[0])

        # Encode the image in base64
        image_data = encode_image(pil_image)

        # Create the prompt
        prompt_template = PromptTemplate(
            input_variables=["image_data", "top", "right", "bottom", "left", "image_width", "image_height", "face_width", "face_height", "landmarks"],
            template="""Analyze the provided facial image and suggest areas where fat injections can be placed. 
            Provide at least 15 points in the following format: 
            [
                {{
                    "region": "Forehead",
                    "coordinates": [150, 100],
                    "meridian": "Gall Bladder (GB)",
                    "acupuncture_point": "GB14",
                    "reason": "Restoring volume to smooth out forehead lines."
                }}
            ]
            Image Data: {image_data}, Bounding Box: Top: {top}, Right: {right}, Bottom: {bottom}, Left: {left}, 
            Width: {image_width}, Height: {image_height}, Face Width: {face_width}, Face Height: {face_height}, 
            Landmarks: {landmarks}
            """
        )

        # Format the prompt
        prompt = prompt_template.format(
            image_data=image_data,
            top=top,
            right=right,
            bottom=bottom,
            left=left,
            image_width=image_width,
            image_height=image_height,
            face_width=face_width,
            face_height=face_height,
            landmarks=formatted_landmarks
        )

        # Call the model and get the response
        response = chain([HumanMessage(content=prompt)])
        response_content = response.content

        fat_injection_data = extract_json_from_string(response_content)

        if fat_injection_data:
            # Annotate the image with the points
            annotated_image = draw_points_on_image(pil_image, fat_injection_data)

            # Convert annotated image to bytes for response
            img_byte_arr = io.BytesIO()
            annotated_image.save(img_byte_arr, format="JPEG")
            img_byte_arr.seek(0)

            return StreamingResponse(img_byte_arr, media_type="image/jpeg")
        else:
            return JSONResponse(content={"error": "No valid data received from model."}, status_code=500)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
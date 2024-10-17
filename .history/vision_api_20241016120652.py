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
        if image_data:
            # Create the prompt template
            prompt_template = PromptTemplate(
                input_variables=["image_data", "top", "right", "bottom", "left", "image_width", "image_height", "face_width", "face_height", "landmarks"],
                template="""
                Analyze the provided facial image and suggest areas where fat injections can be placed to create a more youthful appearance. 
                Use insights from the following references:
                
                - **Bioplasty** by Dr. Almir Nacul
                - **Regenerative Facial Surgery** by Steven R. Cohen, Tunc Tiryaki, Patrick L. Tonnard, and Alexis M. Verpaele
                - **Centrofacial Rejuvenation (Volume III)** by Patrick L. Tonnard, Alexis M. Verpaele, and Richard H. Bensimon

                For each area, provide the following details:

                1. **Name of the region** (e.g., cheeks, under-eye, jawline).
                
                2. Approximate **pixel coordinates (x, y)** of the site based on the extracted facial landmarks.
                
                3. The associated **meridian** that controls that region based on Traditional Chinese Medicine (TCM) (e.g., Gall Bladder (GB), Stomach (St), Bladder (B)).
                
                4. A specific **acupuncture point** from TCM corresponding to that area (e.g., *Facial Beauty (St3)* or *Drilling Bamboo (B2)*).
                
                5. A **reason** explaining why this particular area benefits from fat injections (e.g., restoring volume, enhancing contour, or reducing signs of aging), referencing the techniques and philosophies from the aforementioned books.

                Please provide **a minimum of 15 points** for fat injection sites, but feel free to include more if applicable. 
                Ensure that the response is in valid JSON format.

                Image Data: {image_data}
                Face Bounding Box: Top: {top}, Right: {right}, Bottom: {bottom}, Left: {left}, Face Width: {face_width}, Face Height: {face_height}
                Image Dimensions: Width: {image_width}, Height: {image_height}
                Facial Landmarks: {landmarks}
                
                Sample format:
                [
                    {{
                        "region": "Forehead",
                        "coordinates": [150, 100],
                        "meridian": "Gall Bladder (GB)",
                        "acupuncture_point": "GB14",
                        "reason": "Restoring volume to smooth out forehead lines and enhance a youthful contour, following techniques described in **Regenerative Facial Surgery**."
                    }}
                ]
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

            # Convert annotated image to base64 for the response
            annotated_image_base64 = encode_image(annotated_image)

            return JSONResponse(content={
                "fat_injection_data": fat_injection_data,
                "annotated_image": annotated_image_base64
            })
        else:
            return JSONResponse(content={"error": "No valid data received from model."}, status_code=500)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
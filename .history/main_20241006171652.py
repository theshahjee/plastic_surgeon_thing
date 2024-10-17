import base64
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage

# Function to encode the image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# OpenAI API key
openai_api_key = 'sk-gIMNPrI_kzWMwePo7eIK33WyLcb9YwRAF-8OLupYXoT3BlbkFJLOgRISb6xNI8VN1FGbqHcqy39B6aslj-5VWere7zoA'

# Initialize OpenAI chat model
chain = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", max_tokens=1024)

# Encode the image
image_data = encode_image("./sample_image.jpg")

# Create a prompt template
prompt_template = PromptTemplate(
    input_variables=["image_data"],
    template="""
Analyze the provided facial image and suggest areas where fat injections can be placed to create a more youthful appearance. For each area, provide the following details:

1. **Name of the region** (e.g., cheeks, under-eye, jawline).
  
2. Approximate **pixel coordinates (x, y)** of the site.
  
3. The associated **meridian** that controls that region based on Traditional Chinese Medicine (TCM) (e.g., Gall Bladder (GB), Stomach (St), Bladder (B)).
  
4. A specific **acupuncture point** from TCM corresponding to that area (e.g., *Facial Beauty (St3)* or *Drilling Bamboo (B2)*).
  
5. A **reason** explaining why this particular area benefits from fat injections (e.g., restoring volume, enhancing contour, or reducing signs of aging).

Please provide **a minimum of 15 points** for fat injection sites, but feel free to include more if applicable. The output should return the data structured as **JSON** and include a visualization where these suggested points (and their corresponding labels) are mapped **directly on the facial image**, highlighting the injection sites.
"""
)

# Format the prompt with the image data
prompt = prompt_template.format(image_data=image_data)

# Invoke the model with the prompt
msg = chain.invoke([
    HumanMessage(content=prompt)
])

# Print the response content
response_content = msg.content
print(response_content)

# Extract and store the JSON response
try:
    fat_injection_data = json.loads(response_content)
    # Loop through the data and store each point in variables
    for point in fat_injection_data:
        region = point["region"]
        coordinates = point["coordinates"]
        meridian = point["meridian"]
        acupuncture_point = point["acupuncture_point"]
        reason = point["reason"]

        # Print the stored values (you can modify this to save or use as needed)
        print(f"Region: {region}, Coordinates: {coordinates}, Meridian: {meridian}, Acupuncture Point: {acupuncture_point}, Reason: {reason}")
except json.JSONDecodeError as e:
    print(f"Failed to decode JSON: {e}")

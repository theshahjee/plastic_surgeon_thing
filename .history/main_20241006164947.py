import base64
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
You are a plastic surgeon assistant AI. Analyze the provided facial image and suggest the areas where fat injections should be placed to make the face look younger. 

Please provide the name of each part of the face and the exact (x, y) coordinates for the recommended injection sites.

Image Data: {image_data}

Response format:
- Area: [Name of the face part]
- Coordinates: (x, y)
"""
)

# Format the prompt with the image data
prompt = prompt_template.format(image_data=image_data)

# Invoke the model with the prompt
msg = chain.invoke([
    HumanMessage(content=prompt)
])

# Print the response content
print(msg.content)






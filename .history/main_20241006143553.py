import base64
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
openai_api_key = 'sk-gIMNPrI_kzWMwePo7eIK33WyLcb9YwRAF-8OLupYXoT3BlbkFJLOgRISb6xNI8VN1FGbqHcqy39B6aslj-5VWere7zoA'
chain = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", max_tokens=1024)

image = encode_image("./sample_image.jpg")

prompt = """
Analyze this portrait.Give me your opinion on which parts of the face and which depressions in this face 
I should put fat injections to make this face look younger. Please provide the name of the areas and the coordinates (x, y) of where the fat injections should be placed.
"""

# Invoke the OpenAI API
msg = chain.invoke(
    [
        AIMessage(content=""),  # Preceding context (if needed, otherwise leave blank)
        HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"  # Pass the image as base64
                    },
                },
            ]
        )
    ]
)

# Print the response content
print(msg.content)
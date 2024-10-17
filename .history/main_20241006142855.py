import base64
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
openai_api_key = 'sk-gIMNPrI_kzWMwePo7eIK33WyLcb9YwRAF-8OLupYXoT3BlbkFJLOgRISb6xNI8VN1FGbqHcqy39B6aslj-5VWere7zoA'
chain = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)

image = encode_image("./sample_image.jpg")

msg = chain.invoke(
    [   AIMessage(
        content="You are a useful bot that is especially good at OCR from images"
    ),
        HumanMessage(
            content=[
                {"type": "text", "text": "Identify all items on the this image which are food related and provide a list of what you see"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    },
                },
            ]
        )
    ]
)
print(msg.content)
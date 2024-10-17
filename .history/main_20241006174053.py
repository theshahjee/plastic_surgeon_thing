import base64
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# Replace this with an actual URL to the image if supported
image_url = "https://example.com/sample_image.jpg"

# Initialize OpenAI API key (replace with a secure method)
openai_api_key = 'your-openai-api-key'

# Initialize OpenAI chat model
chain = ChatOpenAI(api_key=openai_api_key, model="gpt-4", max_tokens=1024)

if image_url: 
    # Shorten the prompt by passing a URL instead of the base64 data
    prompt_template = PromptTemplate(
        input_variables=["image_url"],
        template="""
        Analyze the provided facial image from the URL and suggest areas where fat injections can be placed to create a more youthful appearance. For each area, provide the following details:

        1. **Name of the region** (e.g., cheeks, under-eye, jawline).
          
        2. Approximate **pixel coordinates (x, y)** of the site.
          
        3. The associated **meridian** that controls that region based on Traditional Chinese Medicine (TCM) (e.g., Gall Bladder (GB), Stomach (St), Bladder (B)).
          
        4. A specific **acupuncture point** from TCM corresponding to that area (e.g., *Facial Beauty (St3)* or *Drilling Bamboo (B2)*).
          
        5. A **reason** explaining why this particular area benefits from fat injections (e.g., restoring volume, enhancing contour, or reducing signs of aging).

        Please provide **a minimum of 15 points** for fat injection sites, but feel free to include more if applicable. 
        
        Image URL: {image_url}

        Here is a sample response format:

        ```json
        [
            {{
                "region": "Forehead",
                "coordinates": [150, 100],
                "meridian": "Gall Bladder (GB)",
                "acupuncture_point": "GB14",
                "reason": "Restoring volume to smooth out forehead lines and enhance a youthful contour."
            }}
        ]
        ```
        """
    )

    # Format the prompt with the image URL
    prompt = prompt_template.format(image_url=image_url)

    # Invoke the model with the prompt
    response = chain([HumanMessage(content=prompt)])

    # Print the response content
    response_content = response.content
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
else:
    print("Image URL not provided. Exiting program.")

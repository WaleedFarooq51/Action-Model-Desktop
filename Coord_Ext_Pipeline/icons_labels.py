import asyncio
import aiohttp
import base64
import os
import json
import time

# Get the current working directory
CURRENT_DIR = os.getcwd()
Coord_Ext_Pipeline_path = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'Loops-Intel\Coord_Ext_Pipeline'))

# API Keys
api_keys = [
    "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",
    "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",
    "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",
    "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",
    "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",
    "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"
]

# Path to the folder containing images
folder_path = os.path.join(Coord_Ext_Pipeline_path, "extracted-crops")

async def main_process(api_keys, folder_path, model="gpt-4o-mini"):
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Function to process images asynchronously with a specified API key
    async def process_image_with_api_key(session, api_key, img):
        encoded_image = encode_image(os.path.join(folder_path, img))
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be given an image of a UI screen element. It can be an icon or symbol or text. Identify the element or specify what it represents in one or two words. Make sure you do not add any extra unnecessary words other than the specific name or identification of the element itself."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as response:
            response_json = await response.json()
            result = {
                "image_name": img,
                "response": response_json["choices"][0]["message"]["content"].strip()
            }
            return result

    # Function to handle concurrent image processing
    async def process_images_concurrently(image_files, api_keys):
        start_time = time.time()  # Start the timer
        results = []  # List to store the results incrementally
        # JSON file to store the results
        output_file = os.path.join(folder_path, 'image_responses.json')
        # If the file exists, empty it first, else create it
        with open(output_file, 'w') as json_file:
            json_file.write('')
        async with aiohttp.ClientSession() as session:
            tasks = []
            for idx, img in enumerate(image_files):
                api_key = api_keys[idx % len(api_keys)]  # Select API key in a round-robin manner
                task = asyncio.create_task(process_image_with_api_key(session, api_key, img))
                tasks.append(task)
            results = await asyncio.gather(*tasks)
        # Save the final results to the JSON file after all images are processed
        with open(output_file, 'w') as json_file:
            json.dump(results, json_file, indent=4)
        end_time = time.time()  # End the timer
        total_time = end_time - start_time
        print(f"Total time taken with LLM: {total_time:.2f} seconds")
        return total_time, results

    # Get all image files in the folder and sort them in ascending order
    image_files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    # Run the asynchronous image processing
    total_time, results = await process_images_concurrently(image_files, api_keys)
    print(f"All results saved to {os.path.join(folder_path, 'image_responses.json')}")
    return total_time, results


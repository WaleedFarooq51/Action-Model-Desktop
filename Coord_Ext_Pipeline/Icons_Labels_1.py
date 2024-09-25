import openai
import base64
import requests
import os
import json
import time

def process_ui_elements(api_key, folder_path, model="gpt-4o-mini"):
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Get all image files in the folder and sort them in ascending order
    image_files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

    # Function to process images one by one and save results
    def process_images_individually(image_files):
        start_time = time.time()  # Start the timer

        # Initialize an empty list to store results incrementally
        results = []

        # JSON file to store the results
        output_file = os.path.join(folder_path, 'image_responses.json')

        if os.path.exists(output_file):
            # If the file exists, empty it first
            with open(output_file, 'w') as json_file:
                json_file.write('')  # Clear the file content
        else:
            # If the file does not exist, create it
            with open(output_file, 'w') as json_file:
                pass  # This creates the file

        # Now you can write your new data to the file
        with open(output_file, 'w') as json_file:
            json.dump(results, json_file)

        for img in image_files:
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

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            # Parse and store the output
            output = response.json()
            result = {
                "image_name": img,
                "response": output["choices"][0]["message"]["content"].strip()
            }
            results.append(result)

            # Save the updated results to the JSON file after each image is processed
            with open(output_file, 'w') as json_file:
                json.dump(results, json_file, indent=4)

            print(f"Processed and saved result for: {img}")

        end_time = time.time()  # End the timer
        total_time = end_time - start_time
        print(f"Total time taken by LLM to deduce the meaning from the crops is: {total_time:.2f} seconds")

    # Process all images one by one
    process_images_individually(image_files)

    print(f"All results saved to {os.path.join(folder_path, 'image_responses.json')}")


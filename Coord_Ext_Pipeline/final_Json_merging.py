import json
import os

def update_json_content(image_responses_path, final_combined_json_path, output_folder):
    # Load the JSON files
    with open(image_responses_path, 'r') as f:
        image_responses = json.load(f)

    with open(final_combined_json_path, 'r') as f:
        final_combined_json = json.load(f)

    # Create a mapping from image_name to response
    response_mapping = {item['image_name']: item['response'] for item in image_responses}

    # Update the content in final_combined_json based on the mapping
    for component in final_combined_json['compos']:
        image_name = f"{component['id']}.png"
        if image_name in response_mapping:
            component['content'] = response_mapping[image_name]

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, 'PipelineJson.json')

    # Save the new JSON file in the specified folder
    with open(output_file, 'w') as f:
        json.dump(final_combined_json, f, indent=4)

    print(f"New JSON file saved to: {output_file}")


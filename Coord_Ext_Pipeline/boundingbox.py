import os
import cv2
import json

# Get the current working directory
CURRENT_DIR = os.getcwd()

# Define the folders relative to the current directory
PATCHES_FOLDER = os.path.join(CURRENT_DIR, "Before-OCR")
GPT_JSONS_FOLDER = os.path.join(CURRENT_DIR, "GPT-Jsons")
BOUNDING_BOXES_FOLDER = os.path.join(CURRENT_DIR, "Bounding-Box-Patches")

os.makedirs(BOUNDING_BOXES_FOLDER, exist_ok=True)

def draw_bounding_boxes(image, json_data):
    """Draw bounding boxes on the image according to the JSON data."""
    if isinstance(json_data, dict) and "detected_components" in json_data:
        for element in json_data["detected_components"]:
            start_point = (element['column_min'], element['row_min'])
            end_point = (element['column_max'], element['row_max'])
            color = (0, 255, 0)  # Green color for bounding box
            thickness = 2
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
            # Optionally, you can add the label (content) on the bounding box
            label = element["content"]
            cv2.putText(image, label, (element['column_min'], element['row_min'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    else:
        print("Unexpected JSON format. Skipping this image.")
    return image

def process_patches_with_json(patch_name):
    patch_path = os.path.join(PATCHES_FOLDER, f"{patch_name}.png")
    json_path = os.path.join(GPT_JSONS_FOLDER, f"{patch_name}.json")

    # Load the image
    image = cv2.imread(patch_path)

    # Check if the JSON file is empty
    if os.path.getsize(json_path) == 0:
        print(f"JSON file {json_path} is empty. Skipping.")
        return

    # Load the JSON data
    with open(json_path, 'r') as file:
        try:
            json_data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for {patch_name}: {e}")
            return

    # Draw bounding boxes on the image
    image_with_boxes = draw_bounding_boxes(image, json_data)

    # Save the image with bounding boxes in the BOUNDING_BOXES_FOLDER
    bounding_box_image_path = os.path.join(BOUNDING_BOXES_FOLDER, f"{patch_name}.png")
    cv2.imwrite(bounding_box_image_path, image_with_boxes)

    print(f"Processed and saved: {bounding_box_image_path}")

if __name__ == "__main__":
    # Process each patch in PATCHES_FOLDER
    for file_name in os.listdir(PATCHES_FOLDER):
        if file_name.endswith('.png'):
            patch_name = os.path.splitext(file_name)[0]
            json_path = os.path.join(GPT_JSONS_FOLDER, f"{patch_name}.json")
            if os.path.exists(json_path):
                process_patches_with_json(patch_name)
            else:
                print(f"No corresponding JSON file found for {patch_name}. Skipping.")

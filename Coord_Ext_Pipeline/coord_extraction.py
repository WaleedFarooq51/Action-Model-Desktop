from PIL import Image, ImageDraw
from base64 import b64encode
from icons_labels import main_process
#from Icons_Labels_1 import process_ui_elements
from final_Json_merging import update_json_content
from run_single import resize_height_by_longest_edge, detect_components
import requests
import cv2
import os
import shutil
import json
import time
import asyncio

start_time = time.time()

# Get the current working directory
CURRENT_DIR = os.getcwd()
Coord_Ext_Pipeline_path = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'Loops-Intel\Coord_Ext_Pipeline'))

# Define the folders relative to the current directory
PATCHES_FOLDER = os.path.join(Coord_Ext_Pipeline_path, "Before-ip")
AFTER_IP_FOLDER = os.path.join(Coord_Ext_Pipeline_path, "After-ip")
SEPARATE_FILES_FOLDER = os.path.join(Coord_Ext_Pipeline_path, "separate-images-jsons")
AFTER_OCR_FOLDER = os.path.join(Coord_Ext_Pipeline_path, "After-OCR")
FULL_SCREEN_IP_FOLDER = os.path.join(Coord_Ext_Pipeline_path, "Full-Screen")
OCR_IP_FOLDER = os.path.join(Coord_Ext_Pipeline_path, "ocr-&-ip")

# Create necessary directories
os.makedirs(PATCHES_FOLDER, exist_ok=True)
os.makedirs(AFTER_IP_FOLDER, exist_ok=True)
os.makedirs(SEPARATE_FILES_FOLDER, exist_ok=True)
os.makedirs(AFTER_OCR_FOLDER, exist_ok=True)
os.makedirs(FULL_SCREEN_IP_FOLDER, exist_ok=True)
os.makedirs(OCR_IP_FOLDER, exist_ok=True)

# Global variables for rows, columns, and patch height
ROWS = 5
COLUMNS = 10
patch_height = None

def clear_folder(folder_path):
    """Remove all files and subdirectories in the specified folder."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error: {e}")

def load_screenshot(image_path):
    # Load the screenshot from the specified path
    if os.path.exists(image_path):
        image = Image.open(image_path)
        return image
    else:
        raise FileNotFoundError(f"The image at path {image_path} was not found.")

def divide_image_into_patches(image, rows=ROWS, columns=COLUMNS):
    global patch_height  # Declare as global to modify the global variable
    
    # Get the width and height of the image
    img_width, img_height = image.size
    
    # Calculate the size of each patch, ignoring any leftover pixels
    patch_width = img_width // columns
    patch_height = img_height // rows
    
    # Initialize list to store patches
    patches = []
    patch_info = []
    
    # Loop over the rows and columns and create patches
    for row in range(rows):
        for col in range(columns):
            left = col * patch_width
            upper = row * patch_height
            right = left + patch_width
            lower = upper + patch_height
            
            # Crop the patch
            patch = image.crop((left, upper, right, lower))
            patch_id = row * columns + col + 1
            patches.append((patch, patch_id))
            patch_info.append({"patch_id": patch_id, "row": row + 1, "column": col + 1})
    
    return patches, patch_info

def save_patches_and_ids(patches, patch_info, output_dir=PATCHES_FOLDER):
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save each patch with a unique name and store IDs in a JSON file
    ids_json_path = os.path.join(output_dir, 'ids.json')
    for patch, patch_id in patches:
        patch_path = os.path.join(output_dir, f'patch_{patch_id}.png')
        patch.save(patch_path)
        print(f'Saved {patch_path}')
    
    # Save the patch info to a JSON file
    with open(ids_json_path, 'w') as json_file:
        json.dump(patch_info, json_file, indent=4)
    print(f'Saved patch IDs, rows, and columns to {ids_json_path}')

def process_patches(patches):
    key_params = {'min-grad': 10, 'ffl-block': 5, 'min-ele-area': 50,
                  'merge-contained-ele': True, 'merge-line-to-paragraph': False, 'remove-bar': True}
    
    for patch, patch_id in patches:
        patch_name = f'patch_{patch_id}.png'
        patch_path = os.path.join(PATCHES_FOLDER, patch_name)
        
        # Save the patch to disk (if not already saved)
        if not os.path.exists(patch_path):
            patch.save(patch_path)

        # Create a unique directory for each patch inside the AFTER_IP_FOLDER
        patch_folder = os.path.join(AFTER_IP_FOLDER, os.path.splitext(patch_name)[0])
        os.makedirs(patch_folder, exist_ok=True)
        
        # Resize image
        resized_height = resize_height_by_longest_edge(patch_path, resize_length=patch_height)

        # Component detection
        try:
            detect_components(patch_path, patch_folder, key_params, is_clf=False, resized_height=resized_height)
        except IndexError:
            print(f"No components detected in {patch_name}. Saving an empty JSON file.")
            empty_json_path = os.path.join(patch_folder, os.path.splitext(patch_name)[0] + '.json')
            with open(empty_json_path, 'w') as f_out:
                json.dump({"img_shape": list(patch.size), "compos": []}, f_out, indent=4)
        
        # Move the processed image to the 'ip' subfolder, overwriting if necessary
        ip_folder = os.path.join(patch_folder, 'ip')
        os.makedirs(ip_folder, exist_ok=True)
        processed_image_path = os.path.join(ip_folder, patch_name)
        if os.path.exists(processed_image_path):
            os.remove(processed_image_path)
        shutil.copy(patch_path, processed_image_path)  # Copy the original patch to the AFTER_IP_FOLDER without altering the original in BEFORE-IP

def collect_and_copy_files(after_ip_folder, separate_files_folder):
    """Collect image and JSON files from each patch folder and copy them to a separate directory."""
    for patch_folder in os.listdir(after_ip_folder):
        patch_folder_path = os.path.join(after_ip_folder, patch_folder)
        json_outside_ip = os.path.join(patch_folder_path, patch_folder + '.json')
        
        # Check if JSON is outside the 'ip' folder (indicating no components detected)
        if os.path.exists(json_outside_ip):
            destination_json_path = os.path.join(separate_files_folder, patch_folder + '.json')
            shutil.copy(json_outside_ip, destination_json_path)

        # Now, handle the case of images inside the 'ip' folder
        ip_folder_path = os.path.join(patch_folder_path, "ip")
        if os.path.isdir(ip_folder_path):
            for file_name in os.listdir(ip_folder_path):
                source_file_path = os.path.join(ip_folder_path, file_name)
                destination_file_path = os.path.join(separate_files_folder, file_name)
                shutil.copy(source_file_path, destination_file_path)

def remove_png_files(separate_files_folder):
    """Remove all .png files from the separate-files folder."""
    for file_name in os.listdir(separate_files_folder):
        if file_name.endswith('.png'):
            file_path = os.path.join(separate_files_folder, file_name)
            os.remove(file_path)

def combine_jsons(before_ip_folder, separate_images_jsons_folder, original_image_path):
    """Combine individual patch JSONs into a final JSON, ensuring unique IDs across the entire image with scaled coordinates."""
    ids_json_path = os.path.join(before_ip_folder, "ids.json")
    
    # Load ids.json
    with open(ids_json_path, 'r') as ids_file:
        ids_data = json.load(ids_file)

    final_json = {
        "img_shape": [],  # This should reflect the overall image shape
        "compos": []
    }

    unique_id_counter = 1  # Counter to ensure unique IDs across all patches

    # Get the overall dimensions of the image from the first patch's JSON
    first_patch_path = os.path.join(separate_images_jsons_folder, f"patch_1.json")
    with open(first_patch_path, 'r') as first_patch_file:
        first_patch_data = json.load(first_patch_file)
        first_patch_width = first_patch_data["img_shape"][1]
        first_patch_height = first_patch_data["img_shape"][0]

    # Calculate the actual dimensions of the original screenshot
    original_image = Image.open(original_image_path)
    original_width, original_height = original_image.size

    # Calculate total image width and height based on the number of columns and rows
    total_width = first_patch_width * COLUMNS
    total_height = first_patch_height * ROWS

    final_json["img_shape"] = [original_height, original_width]

    # Calculate scaling factors based on the original image dimensions
    width_scale = original_width / total_width
    height_scale = original_height / total_height

    for patch in ids_data:
        patch_id = patch["patch_id"]
        row = patch["row"]
        column = patch["column"]

        # Load the respective JSON file for the patch
        patch_json_path = os.path.join(separate_images_jsons_folder, f"patch_{patch_id}.json")
        
        if not os.path.exists(patch_json_path):
            print(f"Warning: JSON file for patch {patch_id} not found at {patch_json_path}. Skipping this patch.")
            continue
        
        with open(patch_json_path, 'r') as patch_file:
            patch_data = json.load(patch_file)

        # Calculate the offsets based on the current patch's row and column
        x_offset = (column - 1) * first_patch_width
        y_offset = (row - 1) * first_patch_height

        # Modifying the coordinates and ensuring unique IDs with scaling
        for compo in patch_data["compos"]:
            compo["column_min"] = int((compo["column_min"] + x_offset) * width_scale)
            compo["column_max"] = int((compo["column_max"] + x_offset) * width_scale)
            compo["row_min"] = int((compo["row_min"] + y_offset) * height_scale)
            compo["row_max"] = int((compo["row_max"] + y_offset) * height_scale)
            
            # Assign a unique ID to each component
            compo["id"] = unique_id_counter
            unique_id_counter += 1
            
            # Add type "cnn" to the component
            compo["type"] = "cnn"
            
            final_json["compos"].append(compo)

    # Save the final JSON in the full_screen_ip folder
    final_json_path = os.path.join(FULL_SCREEN_IP_FOLDER, "Final_json.json")
    with open(final_json_path, 'w') as final_file:
        json.dump(final_json, final_file, indent=4)

    print(f"Final JSON file saved at: {final_json_path}")

def map_json_to_image(image_path, final_json_path, output_image_path):
    """Map the JSON directly to the image without additional scaling."""
    # Load the original image
    original_image = Image.open(image_path)
    draw = ImageDraw.Draw(original_image)

    # Load the final JSON file
    with open(final_json_path, 'r') as f:
        final_data = json.load(f)

    # Iterate over each component and draw rectangles with the given coordinates
    for compo in final_data["compos"]:
        row_min = compo["row_min"]
        row_max = compo["row_max"]
        col_min = compo["column_min"]
        col_max = compo["column_max"]
        
        # Draw the rectangle on the image
        outline_color = "green" if compo["type"] == "cnn" else "red"
        draw.rectangle([(col_min, row_min), (col_max, row_max)], outline=outline_color, width=2)

    # Save the output image with bounding boxes
    original_image.save(output_image_path)
    print(f"Image with mapped components saved at: {output_image_path}")

def ocr_detection_easyocr(imgpath:str):
    """Perform OCR using EasyOCR."""
    try:
        url = "http://34.135.62.211:8000/process-image"
        response = None
        with open(imgpath, "rb") as file:
            request = {"image_file": (file)}
            response = requests.post(url, files=request)
            response = response.json()
        return response
    except Exception as e:
        print(str(e))
        return None

def ocr_to_custom_json(image_path, json_path):
    print("""Perform OCR using EasyOCR.""")
    
    annotations = ocr_detection_easyocr(image_path)
    
    if annotations:
        print("Detected Text Annotations:")
        
        # Prepare the final data structure
        image = cv2.imread(image_path)
        img_shape = image.shape

        custom_json = {
            "img_shape": [img_shape[0], img_shape[1], img_shape[2]],  # height, width, channels
            "texts": []
        }

        for idx, annotation in enumerate(annotations):
            bounding_box, text, confidence = annotation
            
            column_min = min(point[0] for point in bounding_box)
            column_max = max(point[0] for point in bounding_box)
            row_min = min(point[1] for point in bounding_box)
            row_max = max(point[1] for point in bounding_box)

            text_data = {
                "id": idx,
                "content": text,
                "column_min": int(column_min),
                "row_min": int(row_min),
                "column_max": int(column_max),
                "row_max": int(row_max),
                "type": "ocr"  # Add type "ocr"
            }

            custom_json["texts"].append(text_data)

        # Save the results to a JSON file
        with open(json_path, 'w') as json_file:
            json.dump(custom_json, json_file, indent=4)
        
        print(f"OCR results saved to {json_path}")
    else:
        print("No text detected in the image.")
        # Save an empty JSON if no text is detected
        with open(json_path, 'w') as json_file:
            json.dump({"img_shape": [], "texts": []}, json_file, indent=4)

def draw_bounding_boxes(image_path, json_path, output_image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Load the JSON data
    with open(json_path, 'r') as json_file:
        ocr_data = json.load(json_file)

    # Draw bounding boxes on the image
    for item in ocr_data['texts']:
        # Define the bounding box coordinates
        start_point = (item['column_min'], item['row_min'])
        end_point = (item['column_max'], item['row_max'])
        
        # Draw the rectangle (bounding box)
        image = cv2.rectangle(image, start_point, end_point, (0, 0, 255), 2)  # red color with thickness 2
        
    # Save the annotated image
    cv2.imwrite(output_image_path, image)
    
    print(f"Annotated image saved as {output_image_path}")

def remove_cnn_under_ocr(cnn_json_path, ocr_json_path):
    """Remove CNN bounding boxes that are under OCR bounding boxes and replace them with OCR entries."""
    
    # Load the CNN and OCR JSON data
    with open(cnn_json_path, 'r') as cnn_file:
        cnn_data = json.load(cnn_file)
    
    with open(ocr_json_path, 'r') as ocr_file:
        ocr_data = json.load(ocr_file)
    
    combined_data = {
        "img_shape": cnn_data["img_shape"],
        "compos": []
    }

    unique_id_counter = 1  # Ensure unique IDs
    
    # Iterate over each OCR bounding box
    for ocr_item in ocr_data["texts"]:
        ocr_bbox = (ocr_item["column_min"], ocr_item["row_min"], ocr_item["column_max"], ocr_item["row_max"])
        
        # Filter out CNN components that overlap with the OCR bounding box
        cnn_data["compos"] = [
            compo for compo in cnn_data["compos"]
            if not (
                compo["column_min"] >= ocr_bbox[0] and compo["column_max"] <= ocr_bbox[2] and
                compo["row_min"] >= ocr_bbox[1] and compo["row_max"] <= ocr_bbox[3]
            )
        ]
        
        # Add the OCR bounding box to the combined data
        combined_data["compos"].append({
            "id": unique_id_counter,
            "content": ocr_item["content"],
            "column_min": ocr_item["column_min"],
            "row_min": ocr_item["row_min"],
            "column_max": ocr_item["column_max"],
            "row_max": ocr_item["row_max"],
            "type": "ocr"  # Add type "ocr"
        })
        unique_id_counter += 1
    
    # Add remaining CNN components to the combined data
    for compo in cnn_data["compos"]:
        compo["id"] = unique_id_counter  # Ensure unique IDs
        compo["type"] = "cnn"  # Add type "cnn"
        combined_data["compos"].append(compo)
        unique_id_counter += 1
    
    return combined_data

def save_final_json_and_image(image_path, combined_data, output_json_path, output_image_path):
    """Save the final combined JSON and map it onto the original image."""
    
    # Save the final JSON
    with open(output_json_path, 'w') as json_file:
        json.dump(combined_data, json_file, indent=4)
    
    # Load the original image
    original_image = Image.open(image_path)
    draw = ImageDraw.Draw(original_image)
    
    # Draw bounding boxes from the combined data
    for compo in combined_data["compos"]:
        outline_color = "green" if compo["type"] == "cnn" else "red"
        draw.rectangle(
            [(compo["column_min"], compo["row_min"]), (compo["column_max"], compo["row_max"])],
            outline=outline_color, width=2
        )
    
    # Save the mapped image
    original_image.save(output_image_path)


def coordinates_extraction(screenshot):
    print("Coordinates Extraction Process Started")
    # Clear the folders before saving new patches and results
    clear_folder(PATCHES_FOLDER)
    clear_folder(AFTER_IP_FOLDER)
    clear_folder(SEPARATE_FILES_FOLDER)
    clear_folder(AFTER_OCR_FOLDER)
    clear_folder(FULL_SCREEN_IP_FOLDER)
    clear_folder(OCR_IP_FOLDER)
    
    # Define the image path relative to the current directory
    IMAGE_PATH = screenshot
    
    # Load the screenshot
    screenshot = load_screenshot(IMAGE_PATH)
    
    # Divide the image into patches and get their IDs and positions
    patches, patch_info = divide_image_into_patches(screenshot)
    
    # Save the patches and their IDs
    save_patches_and_ids(patches, patch_info)
    print(f"Image divided into {len(patches)} patches and stored in {PATCHES_FOLDER}.")
    
    # Process each patch with component detection
    process_patches(patches)
    print(f"Component detection completed. Results are stored in {AFTER_IP_FOLDER}.")
    
    # Collect image and JSON files from each patch and store them in the separate-files folder
    collect_and_copy_files(AFTER_IP_FOLDER, SEPARATE_FILES_FOLDER)
    
    # Remove any .png files from the separate-files folder
    remove_png_files(SEPARATE_FILES_FOLDER)

    # Combine JSON files into a final JSON with unique IDs and scaled coordinates
    FINAL_JSON_PATH = os.path.join(FULL_SCREEN_IP_FOLDER, "Final_json.json")
    combine_jsons(PATCHES_FOLDER, SEPARATE_FILES_FOLDER, IMAGE_PATH)

    # Map the final JSON back to the original image and save the result
    OUTPUT_IMAGE_PATH = os.path.join(FULL_SCREEN_IP_FOLDER, "Mapped_Pannel.png")
    map_json_to_image(IMAGE_PATH, FINAL_JSON_PATH, OUTPUT_IMAGE_PATH)

    # OCR Processing and Saving Results
    OCR_JSON_PATH = os.path.join(AFTER_OCR_FOLDER, "ocr_output.json")
    OCR_ANNOTATED_IMAGE_PATH = os.path.join(AFTER_OCR_FOLDER, "OCR-Image.jpg")
    
    # Perform OCR and save results to JSON
    ocr_to_custom_json(IMAGE_PATH, OCR_JSON_PATH)

    # Draw bounding boxes on the image based on OCR results and save the annotated image
    draw_bounding_boxes(IMAGE_PATH, OCR_JSON_PATH, OCR_ANNOTATED_IMAGE_PATH)

    # Combine the CNN and OCR JSON data, removing overlapping CNN components
    FINAL_COMBINED_JSON_PATH = os.path.join(OCR_IP_FOLDER, "Final_combined_json.json")
    FINAL_COMBINED_IMAGE_PATH = os.path.join(OCR_IP_FOLDER, "Mapped_ocr_ip_Image.png")
    
    combined_data = remove_cnn_under_ocr(
        cnn_json_path=FINAL_JSON_PATH,
        ocr_json_path=OCR_JSON_PATH
    )
    
    # Save the final combined JSON and image
    save_final_json_and_image(IMAGE_PATH, combined_data, FINAL_COMBINED_JSON_PATH, FINAL_COMBINED_IMAGE_PATH)

    print(f"Final combined JSON and mapped image saved in {OCR_IP_FOLDER}.")

    # Integrate the new code here
    # Load the JSON data
    with open(FINAL_COMBINED_JSON_PATH, 'r') as f:
        data = json.load(f)

    # Open the original image
    image = Image.open(IMAGE_PATH)

    # Directory to save cropped images
    output_dir = os.path.join(Coord_Ext_Pipeline_path, "extracted-crops")
    # Check if the directory exists and is not empty
    if os.path.exists(output_dir) and os.listdir(output_dir):
        # If the directory is not empty, clear it
        shutil.rmtree(output_dir)

    # Create the directory (again) after clearing it or if it didn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each element in the JSON
    for element in data['compos']:
        # Extract coordinates and expand by 10 pixels
        row_min = max(element['row_min'] - 20, 0)
        row_max = min(element['row_max'] + 10, image.size[1])
        column_min = max(element['column_min'] - 10, 0)
        column_max = min(element['column_max'] + 10, image.size[0])
        
        # Crop the image
        crop = image.crop((column_min, row_min, column_max, row_max))
        
        # Save the cropped image
        crop_path = os.path.join(output_dir, f"{element['id']}.png")
        crop.save(crop_path)

    print("Crops are made and saved.")

    api_keys = [
    "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",
    "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",
    "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",
    "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",
    "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",
    "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"
    ]
    api_key = "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"

    folder_path_extracted = output_dir

    # Call the main_process function
    total_time, results = asyncio.run(main_process(api_keys, folder_path_extracted))
    #process_ui_elements(api_key=api_key, folder_path=folder_path_extracted)
    print("Meaning has been deduced from the LLM and are save in the Json file.")

    # Directory to save the final JSON file
    final_json_folder = os.path.join(Coord_Ext_Pipeline_path, "Final Json")

    image_responses_path1 = os.path.join(output_dir, 'image_responses.json')
    final_combined_json_path = os.path.join(OCR_IP_FOLDER, 'Final_combined_json.json')
    update_json_content(image_responses_path1, final_combined_json_path, final_json_folder)
    
    # Call the update_json_content function
    image_responses_path = os.path.join(output_dir, 'image_responses.json')
    final_combined_json_path = FINAL_COMBINED_JSON_PATH
    update_json_content(image_responses_path, final_combined_json_path, final_json_folder)

    print("Final JSON from this pipeline is made.")
    print(" ")

    end_time = time.time()

    # Calculate and print the total time taken
    total_time = end_time - start_time
    print(f"Total time taken by the whole pipeline: {total_time:.2f} seconds")














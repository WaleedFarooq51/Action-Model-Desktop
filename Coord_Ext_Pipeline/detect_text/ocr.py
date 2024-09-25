import pytesseract
from PIL import Image

def tesseract_ocr(imgpath):
    """
    Perform OCR using Tesseract on the given image file.

    Parameters:
        imgpath (str): Path to the image file.

    Returns:
        list: Extracted text data including bounding box information.
    """
    image = Image.open(imgpath)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    text_data = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 0:  # Confidence level
            text_data.append({
                "description": data['text'][i],
                "boundingPoly": {
                    "vertices": [
                        {"x": data['left'][i], "y": data['top'][i]},
                        {"x": data['left'][i] + data['width'][i], "y": data['top'][i]},
                        {"x": data['left'][i] + data['width'][i], "y": data['top'][i] + data['height'][i]},
                        {"x": data['left'][i], "y": data['top'][i] + data['height'][i]}
                    ]
                }
            })
    return text_data

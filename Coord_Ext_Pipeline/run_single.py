from os.path import join as pjoin
import cv2
import os
import numpy as np

def resize_height_by_longest_edge(img_path, resize_length):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))

def perform_ocr(input_path_img, output_root):
    import detect_text.text_detection as text
    os.makedirs(pjoin(output_root, 'ocr'), exist_ok=True)
    text.text_detection(input_path_img, output_root, show=False, method='google')

def detect_components(input_path_img, output_root, key_params, is_clf=False, resized_height=None):
    import detect_compo.ip_region_proposal as ip
    os.makedirs(pjoin(output_root, 'ip'), exist_ok=True)
    classifier = None
    if is_clf:
        classifier = {}
        from cnn.CNN import CNN
        classifier['Elements'] = CNN('Elements')
    ip.compo_detection(input_path_img, output_root, key_params,
                       classifier=classifier, resize_by_height=resized_height, show=False)

def merge_results(input_path_img, output_root, key_params):
    import detect_merge.merge as merge
    os.makedirs(pjoin(output_root, 'merge'), exist_ok=True)
    name = os.path.basename(input_path_img).split('.')[0]
    compo_path = pjoin(output_root, 'ip', f'{name}.json')
    ocr_path = pjoin(output_root, 'ocr', f'{name}.json')
    merge.merge(input_path_img, compo_path, ocr_path, pjoin(output_root, 'merge'),
                is_remove_bar=key_params['remove-bar'], is_paragraph=key_params['merge-line-to-paragraph'], show=False)
    
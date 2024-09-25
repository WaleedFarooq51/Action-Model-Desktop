import sys
import os

# Add the path to the 'Coord Extraction Pipeline' folder to Current working directory path
Coord_Ext_Pipeline_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Loops-Intel\Coord_Ext_Pipeline'))
sys.path.append(Coord_Ext_Pipeline_path)

from openai import  OpenAI
from groq import Groq
from Loops_action import subtask_execution
from collections import OrderedDict
from coord_extraction import coordinates_extraction 
import numpy as np
import pyautogui
import json
import re
import time

openai_client = OpenAI(api_key= 'sk-proj-NQBeE2RkXWcGIsOvjssMT3BlbkFJKZlzrxKv6506qa2d68UB')    #Asaad's Key

openai_client_1 = OpenAI(api_key= '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')  #GPT4-o's Key

groq_llama3_client = Groq(api_key='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')

## Task Decomposer
def task_decomposer(user_query):

    task_decomposing_instructions= """
You are Task decomposition agent. You are an important part of an Artificial General Intelligence where your primary function is to generate an action plan for the given user query.
You will be given a user query as input and your output should be an action plan.
Understand the user's core objective of the user query, decompose that user query and generate an action plan at very granular level.
Your operations include but are not limited to clicking buttons, typing information.
Your decomposed Task action plan should be in the following template:
Task Decomposition Agent:
1. Identify the user's objective: You should understand the core objective of user's task.
2. Extract Key Information: Extract all key information that is given by the user.
3. Determine Constraints and Preferences: Identify any potential constraints and challenges that can be occured during task decomposition.
4. Decide on Data Sources: Provide required data sources that is necessary for task execution.
5. Formulate Action Plan: Provide an action plan at a very granular level, focusing exclusively on granular actions necessary to achieve the task. 

Instructions for Formulate Action Plan:
    - Steps must be at very granular level. The task must be decomposed in to very small and granualr steps.
    - Steps must involve only continuous, active engagements and inputs that move the process forward without delay or reviewing.
    - Each step should be proactive and contribute directly to task progression. Exclude any passive activities, such as waiting for system responses or reviewing any details.

Decomposer Limitations:
- You donot have the ability to discuss anything, you can only create plans.
- You donot have the ability to click, move or place the mouse cursor to anywhere on the screen.
- You donot have the ability to click, move or place the mouse cursor inside the text area. 
- you donot have the ability to review anything.
- you donot have the ability to execute actions.
- you donot have the abilty to verify anything.
- you donot have the ability to confirm anything.
- you donot have the ability to negotiate anything.

Following are the examples for your better understanding.
Example 1:
Query: Type ahmed in Notepad, select the text and save the file
Task Decomposition Agent:
1. Identify the Task: The task is to type ahmed and save a file in Notepad.
2. Extract Key Information:
  - Application: Notepad
3. Determine Constraints and Preferences: No specific constraints or preferences are mentioned.
4. Decide on Data Sources: To open a new file, one would typically use a Notepad application.
5. Formulate Action Plan:
  - Open Notepad.
  - Type ahmed
  - Select the text ahmed by using the keyboard shortcut Ctrl+A.
  - Click on File.
  - Click on Save.

Example 2:
Query: Insert the image in word and save that document
Task Decomposition Agent:
1. Identify the user's objective: The task is to insert an image in a Word document and save the document.
2. Extract Key Information:
  - Application: Word
  - Task: Insert image
  - Task: Save document
3. Determine Constraints and Preferences: No specific constraints or preferences are mentioned.
4. Decide on Data Sources:
  - Word application
  - Image file to be inserted
5. Formulate Action Plan:
  - Open Word.
  - Click on Blank document to open a new document.
  - Click on the "Insert" tab.
  - Click on "Pictures" in the "Illustrations" group.
  - Select the image file to be inserted.
  - Click "Insert" to insert the image.
  - Place the image at the desired location in the document.
  - Click on the "File" tab.
  - Click on "Save As" or "Save" depending on whether the document is new or already exists.
  - Choose a location to save the document.
  - Enter a file name.
  - Click "Save" to save the document.

Example 3:
Query: Type ali in word, make it bold and save the document
Task Decomposition Agent:
1. Identify the user's objective: The task is to type ali in a Word document and save the document.
2. Extract Key Information:
  - Application: Word
  - Task: Type ali
  - Task: Save document
3. Determine Constraints and Preferences: No specific constraints or preferences are mentioned.
4. Decide on Data Sources:
  - Word application
5. Formulate Action Plan:
  - Open Word.
  - Click on Blank document to open a new document.
  - Type ali in the text area.
  - Select the text ali by using the keyboard shortcut Ctrl+A.
  - Click on the Home tab.
  - Click on the Bold button in the Font group.
  - Click on the File tab.
  - Select the image file to be inserted.
  - Click "Insert" to insert the image.
  - Place the image at the desired location in the document.
  - Click on the "File" tab.
  - Click on "Save As" or "Save" depending on whether the document is new or already exists.
  - Choose a location to save the document.
  - Enter a file name.
  - Click "Save" to save the document.

Note: When the plan is created donot say lets execute it at the end we dont need it.
    """

    response = groq_llama3_client.chat.completions.create(
        model= "llama3-70b-8192",
        temperature=0.1,
        messages= [
            {"role": "system", "content": task_decomposing_instructions+ "\n" + """Consider yourself as Task Execution agent. Never say that you cannot work on the desktop applications."""},
            {"role": "user", "content": user_query} ],)
    return response.choices[0].message.content

user_query= input("Enter the Query: ")

decomposer= task_decomposer(user_query)
#print(decomposer)

header = "5. Formulate Action Plan:"
start_index = decomposer.find(header) + len(header)
subtasks = decomposer[start_index:].strip()
action_plan = subtasks.strip().split('\n')
action_plan = [subtask.replace('"', '') for subtask in action_plan]
print("Action Plan: ",action_plan)
print("----------------------------------------------------------------")

## Refining coordinates JSON 
def refine_json():
 ## Coordinates JSON File
 json_file_path = os.path.join(Coord_Ext_Pipeline_path, "Final Json", "PipelineJson.json")

 with open(json_file_path, 'r') as file:
    json_file= json.load(file, object_pairs_hook=OrderedDict)

 ## Refining JSON File
 for item in json_file.get('compos', []):          # Remove the "class", "width", and "height" keys 
    item.pop('class', None) 
    item.pop('width', None)
    item.pop('height', None)

    # Calculate x_value and y_value
    x_value = (item['column_min'] + item['column_max']) / 2
    y_value = (item['row_min'] + item['row_max']) / 2
    
    # Add the calculated values to the json file
    item['x_value'] = x_value
    item['y_value'] = y_value

    # Remove the "min" and "max" keys
    item.pop('column_min', None)
    item.pop('column_max', None)
    item.pop('row_min', None)
    item.pop('row_max', None)

 json_file.pop('img_shape', None)
 json_file['ui_elements'] = json_file.pop('compos')

 # Save the refined data back to a JSON file
 with open(os.path.join(Coord_Ext_Pipeline_path, "Final Json", "Refined_Json_file.json"), 'w') as file:
  json.dump(json_file, file, indent=4)
  

 # Read the refined JSON file as a string
 with open(os.path.join(Coord_Ext_Pipeline_path, "Final Json", "Refined_Json_file.json"), 'r') as file:
  json_data = file.read()

  return json_data
    
## JSON Creation for Action execution
def json_creation(plan,x,y):
    
    user_prompt= """
    
Action:
----------------
$action

X Value:
----------------
$x_value

Y Value:
----------------
$y_value

    """

    json_instructions= """
You will receive only one action at a time and x and y values. Your task is to convert this action into a structured JSON format. Each action and its x and y values should be mapped into a corresponding JSON object based on the following template:

{
  'Subtask': [
    {
      'name': 'Open Application',
      'action': 'open_application',
      'parameters': {
        'application': 'application.exe'
      }
    }
  ]
}

{
  'Subtask': [
    {
      'name': 'Type text',
      'action': 'type_text',
      'parameters': {
        'text': 'ali',
        'delay': 2
      }
    }
  ]
}

{
  'Subtask': [
    {
      'name': 'Click on Element',
      'action': 'click_coordinates',
      'parameters': {
        'element_type': 'button',
        'element_name': 'Element name',
        'x': x value,
        'y': y value,
        'delay': 1
      }
    }
  ]
}

Instructions:

1. For each high-level action, create a subtask with the following structure:
   - "name": A descriptive name for the action.
   - "action": The specific action being performed.
   - "parameters": A dictionary of relevant parameters, such as coordinates, element names, and delays.

2. Ignore duplicate or redundant steps and focus on unique actions.

3. For task involving open application, you must only provide structured json format for opening application task. No extra action is required here.

Following are the examples for your better understanding.

Example 1:
{
  'Subtask': [
    {
      'name': 'Open Notepad',
      'action': 'open_application',
      'parameters': {
        'application': 'Notepad.exe'
      }
    }
  ]
}

Example 2:
{
  'Subtask': [
    {
      'name': 'Open word',
      'action': 'open_application',
      'parameters': {
        'application': 'winword.exe' 
      }
    }
  ]
}

Example 2:
{
  'Subtask': [
    {
      'name': 'Open settings',
      'action': 'open_application',
      'parameters': {
        'application': 'ms-settings:' 
      }
    }
  ]
}

Example 3:
{
  'Subtask': [
    {
      'name': 'Click on File',
      'action': 'click_coordinates',
      'parameters': {
        'element_type': 'button',
        'element_name': 'File',
        'x': 100,
        'y': 200,
        'delay': 1
      }
    }
  ]
}

Example 4:
{
  'Subtask': [
    {
        "name": "Right-click in the desired location",
        "action": "right_click",
        "parameters": {
            "element_type": "button",
            "element_name": "desired location",
            "x": 300,
            "y": 400,
            "delay": 1
        }
    }
  ]
}

Example 5:
{
  'Subtask': [
    {
      'name': 'Choose a location to save the file (e.g., Desktop)',
      'action': 'click_coordinates',
      'parameters': {
        'element_type': 'button',
        'element_name': 'Desktop',
        'x': 300,
        'y': 500,
        'delay': 1
      }
    }
  ]
}

Example 6:
{
  'Subtask': [
    {
      'name': 'Select the image file to be inserted',
      'action': 'click_coordinates',
      'parameters': {
        'element_type': 'image file',
        'element_name': 'image file',
        'x': 300,
        'y': 500,
        'delay': 1
      }
    }
  ]
}

Example 7:
{
  'Subtask': [
    {
        "name": "Locate the Notepad file to be copied",
        "action": "click_coordinates",
        "parameters": {
            "element_type": "button",
            "element_name": "Notepad file",
            "x": 300,
            "y": 400,
            "delay": 1
        }
    }
  ]
}

Example 8:
{
  'Subtask': [
    {
        "name": "Navigate to the Desktop",
        "action": "click_coordinates",
        "parameters": {
            "element_type": "button",
            "element_name": "Desktop",
            "x": 300,
            "y": 400,
            "delay": 1
        }
    }
  ]
}

Example 9:
{
  'Subtask': [
    {
        "name": "Type Zain ",
        "action": "type_text",
        "parameters": {
            "text": "Zain",
            "delay": 1
         }
    }
  ]
}

Example 10:
{
  'Subtask': [
    {
        "name": "Type asd",
        "action": "type_text",
        "parameters": {
            "text": "asd",
            "delay": 1
         }
    }
  ]
}

Example 11:
{
  'Subtask': [
    {
        "name": "Select the text by using the keyboard shortcut Ctrl+a",
        "action": "select_text",
        "parameters": {
            "key1": "ctrl"
            "key2": "a"
        }
    }
  ]
}

Example 12:
{
  'Subtask': [
    {
        "name": "Enter a file name",
        "action": "type_text",
        "parameters": {
                'text': 'file.txt',
                'delay': 2
        }
    }
  ]
}
    """

    user_prompt = user_prompt.replace("$action", plan) 
    user_prompt = user_prompt.replace("x_value", x)
    user_prompt = user_prompt.replace("$y_value", y)

    response = openai_client.chat.completions.create(
        model= "gpt-3.5-turbo",
        temperature=0.1,
        messages= [
            {"role": "system", "content": json_instructions},
            {"role": "user", "content": user_prompt} ],)
    return response.choices[0].message.content

def extract_json(response,filename_index):
    # Use regular expression to extract JSON part
    json_match = re.search(r'\{.*\}', response, re.DOTALL)

    if json_match:
        json_str = json_match.group(0)
        json_str = json_str.replace("'", '"')

        try:
            # Parse the extracted JSON string to ensure it's valid
            parsed_json = json.loads(json_str)
            formatted_json = json.dumps(parsed_json, indent=4)

            # Save the formatted JSON to a file
            with open(f'extracted_task_{filename_index}.json', 'w') as json_file:
                json.dump(parsed_json, json_file, indent=4)
            return formatted_json
        
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

    else:
        print("No valid JSON found in the response.")

## Subtask recommendation
def subtask_recom(subtask,json_data,i):

    user_prompt= """
    
Subtask:
----------------
$subtask

JSON Data :
----------------
$json_data
    """

    subtask_recom_instructions= """You are a Subtask Recommendation Agent, designed to assist in automating UI tasks by analyzing and matching subtasks to specific UI elements based on their labels.
You will be provided with two inputs. The first input is a subtask and the second input is data in the form of JSON. 
The subtask is a single task taken from an action plan that was created by a decomposer. 
The data in the form of JSON consists of following entries:
-id: The unique id of each entry.
-content: The label of the UI element.

Your primary task is to carefully analyze the `content` in each JSON entry and determine which UI element most closely matches the provided subtask. 
The recommendation should focus on finding the UI element that directly correlates with the action described in the subtask.
Instructions:
 - Carefully read the subtask and understand what action it requires.
 - Review each entry in the JSON data, particularly the `content` field, to find the UI element that best matches the subtask.
 - Ignore any elements that do not relate to the subtask or do not contain relevant labels.
 - Your recommendation should focus on accuracy: choose the UI element whose `content` best fits the subtask description.
Your output should be in the following format:
- ID: [ID of the recommended UI element]
- Content: [Label of the recommended UI element]
- X Value: [X-coordinate of the recommended UI element]
- Y Value: [Y-coordinate of the recommended UI element]

Following are the examples for your better understanding.
Example 1: 
Subtask: Click save to save the file.
JSON data: {
             "id": 2,
             "content": "Save",
             "type": "ocr",
             "x_value": 30.0,
             "y_value": 63.0
           }
Subtask Recommendation Agent:
-ID: 2
-Content: Save
-X Value: 30.0
-Y Value  63.0

Note: Do not provide the response in the steric format.
    """

    print("Current Subtask: ",subtask)

    if i==0:
     # JSON Response Action 
     json_created= json_creation(subtask, " ", " ")
     json_response= extract_json(json_created,i)
     print("JSON is created.") 

    else: 
     user_prompt = user_prompt.replace("$subtask", subtask) 
     user_prompt = user_prompt.replace("$json_data", json_data) 
    
     response = openai_client_1.chat.completions.create(
        model= "gpt-4o",
        temperature=0.1,
        messages= [
            {"role": "system", "content": subtask_recom_instructions},
            {"role": "user", "content": user_prompt} ],)
     recom= response.choices[0].message.content
     print("Recommended ID is: ",recom)
     
     # Use a regular expression to find all recommended X and Y values 
     x_value_match = re.compile(r'X Value: ([\d.]+)').search(recom)
     y_value_match = re.compile(r'Y Value: ([\d.]+)').search(recom)

     x_value = x_value_match.group(1) if x_value_match else None
     y_value = y_value_match.group(1) if y_value_match else None
     print(f"X Value: {x_value}")
     print(f"Y Value: {y_value}")

     # JSON Response Action 
     json_created= json_creation(subtask, x_value, y_value)
     json_response= extract_json(json_created,i)
     print("JSON is created.")

## Maintaining Screenshots
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def images_are_different(img1, img2, threshold=300):
    npimg1 = np.array(img1)
    npimg2 = np.array(img2)
    error = mse(npimg1, npimg2)
    return error > threshold

def is_unique(current, previous_list):
            for prev in previous_list:
                if not images_are_different(current, prev):
                    return False
            return True

## main function
def main():

 ## For single screenshot:
 #coordinates_extraction(os.path.join(Coord_Ext_Pipeline_path, "Image", "word.png"))

 screenshots_list =[]

 for i in range(len(action_plan)):
  current_subtask = action_plan[i]

  if i==0:
   ## Function for action recom
   subtask_recom(current_subtask," ",i)

   ## Function to execute tasks
   subtask_execution(f'extracted_task_{i}.json')
   time.sleep(5)                                  # Adding a small delay between subtasks for stability
   print("----------------------------------------------------------------")
 
  else:
  #Taking screenshot
   screenshot_path= os.path.join(Coord_Ext_Pipeline_path, "Image", f"screenshot{i}.png")
   screenshot = pyautogui.screenshot()

   if is_unique(screenshot, screenshots_list):
    screenshot.save(screenshot_path)
    screenshots_list.append(screenshot)
    print("Screenshot Saved")

    ## Function for coord extraction
    coordinates_extraction(screenshot_path)

    ## Function for refine JSON
    json_data= refine_json()

   ## Function for action recom
   subtask_recom(current_subtask,json_data,i)

   ## Function to execute tasks
   subtask_execution(f'extracted_task_{i}.json')
   time.sleep(5)                                  # Adding a small delay between subtasks for stability
   print("----------------------------------------------------------------")

if __name__ == "__main__":
    main()
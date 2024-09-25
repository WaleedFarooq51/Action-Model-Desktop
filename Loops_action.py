import json
import pyautogui
import time

## Function to open the application
def open_application(application):     
    # Open the Run dialog
    pyautogui.hotkey('win', 'r')
    time.sleep(1)  # Wait for the Run dialog to open
    
    # Type the application name
    pyautogui.write(application, interval=0.1)
    pyautogui.press('enter')
    time.sleep(2)  # Wait for the application to open
    
    # Maximize the application
    pyautogui.hotkey('win', 'up')

## Function to write text
def type_text(text):
    pyautogui.write(text, interval=0.1)

## Function to click any button
def click_coordinates(x, y):
    pyautogui.moveTo(x, y)
    pyautogui.click(x, y)

## Function to right click any button
def right_click_coordinates(x, y):
    pyautogui.rightClick(x, y)

## Function to select text
def select_text(key1, key2):
    pyautogui.hotkey(key1, key2)

## Function to perform actions
def perform_subtask(subtask):
    action = subtask["action"]
    parameters = subtask["parameters"]

    if action == "open_application":
        open_application(parameters["application"])

    elif action == "type_text":
        type_text(parameters["text"])

    elif action.startswith("click_coordinates"):
        x = parameters["x"]
        y = parameters["y"]
        click_coordinates(x, y)

    elif action == "right_click":
        x = parameters["x"]
        y = parameters["y"]
        right_click_coordinates(x, y)

    elif action == "select_text":
        key1 = parameters["key1"]
        key2 = parameters["key2"]
        select_text(key1,key2)

    else:
        print(f"Unknown action: {action}")

## Function to pass the Action execution Json file
def subtask_execution(task_file):
    with open(task_file, 'r') as file:
        data = json.load(file)

    subtasks = data["Subtask"][0]

    print(f"Performing subtask: {subtasks['name']}")
    perform_subtask(subtasks)
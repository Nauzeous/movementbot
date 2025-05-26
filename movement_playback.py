import json
import time
import keyboard  

with open('generated_movement.json', 'r') as f:
    data = json.load(f)

sequence = data['generated_sequence']

print("Starting input replay in 5 seconds...")
time.sleep(5)

for entry in sequence:
    keys = entry['key'].split('+')  
    duration = entry['duration']
    for k in keys:
        if k == 'None':
            break
        keyboard.press(k)
        print("pressing ",k)

    time.sleep(duration)


    for k in keys:
        if k == 'None':
            break
        keyboard.release(k)

print("Finished replaying sequence.")

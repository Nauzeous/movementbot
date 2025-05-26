from pynput import keyboard
import time
import json

active_keys = []
sequences = []
current_sequence = []
recording = False
effective_dir = ["None", 0.0]

def update_effective_dir():
    global current_sequence, recording, effective_dir
    t = time.time()
    
    if effective_dir[1] > 0:  
        duration = round(t - effective_dir[1], 3)


        if duration > 0.0 and not (effective_dir[0] == 'None' and duration > 1.0):
             current_sequence.append({
                "key": effective_dir[0],
                "duration": duration
            })
    
    # end sequence if over 10 keys long
    if not active_keys and len(current_sequence) > 10:
        sequences.append({"sequence_id": len(sequences) + 1, "actions": current_sequence})
        current_sequence = []
        recording = False
        print("Sequence recorded. Press keys to start a new sequence.")

    hor = 1 
    ver = 1  
    
    for k in active_keys:
        match k:
            case 'a':
                hor -= 1
            case 'd':
                hor += 1
            case 'w':
                ver -= 1
            case 's':
                ver += 1
    
    
    effective_dir[0] = [
        ["w+a","w"   ,"w+d"], 
        ["a"  ,"None","d"  ],
        ["s+a","s"   ,"s+d"]][ver][hor]

    effective_dir[1] = t

def on_press(key):
    global recording, active_keys
    
    try:
        # try and get char from key
        key_char = key.char
    except AttributeError: # special key
        return  
    
    if key_char not in active_keys:
        active_keys.append(key_char)
        
        if not recording:
            recording = True
            effective_dir[1] = time.time()  
            print("Recording started...")
        
        update_effective_dir()

def on_release(key):
    global recording, current_sequence, active_keys
    
    try:
        key_char = key.char
    except AttributeError:
        if key == keyboard.Key.tab:  # tab is used to save and quit
            update_effective_dir()
            sequences.append({"sequence_id": len(sequences) + 1, "actions": current_sequence})
            save_sequences()
        return
    
    if key_char in active_keys:
        active_keys.remove(key_char)
        update_effective_dir()
        
def save_sequences():
    with open("movement_data.json", "w") as f:
        json.dump(sequences, f, indent=2)
    print(f"Saved {len(sequences)} sequences to movement_data.json")


with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    print("Press keys to start recording. Press tab to save and quit.")
    listener.join()
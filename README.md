The movementbot is a neural network that generates movement sequences, trained on actual human movement

It uses the last 10 or so moves and uses an attention layer and a GRU layer to output a key:duration pair in json

Usage:
1.
use the movement_rec.py to record movements
it's a basic movement recorder, use tab to end recording
it records the actual direction, so pressing a + d is equal to standing still
any period of standing still >1s is not recorded, so you dont have to worry about constantly moving during the recording
your data should be a few minutes long

2. you will have a generated movement_data.json file, now you can run the main.py file to create the .pth and generated movement
if you want you can adjust the length of the generated sequence
i occasionally had issues with the generated sequence where all the durations are 0.0s, in which case just run it again i cba to fix

3.
use the movement_playback.py file to play the generated_movement.json file as a sequence of key presses

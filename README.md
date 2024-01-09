# Automated_door_system
In this project we created automated door using face recogination and arduino.
## Steps:
1. Create Environment
2. Install Requirements
3. Setup the circuit and "upload face_recogination_cpmplete.ino" in arduino usind arduino ide
![Circuit](circuit_diagram.png)
4. Select the port address of arduino in faces.py
'''python 
arduino = serial.Serial(port='COM3', baudrate=9600, timeout=.1)
'''
5. Store the images in "images" folder according to names
6. Run "Face_trainer.py" and then "face.py"
source ~/env_buah/bin/activate
pip install RPi.GPIO RPLCD smbus2

import cv2
import os
import sys
import RPi.GPIO as GPIO
from RPLCD.i2c import CharLCD
from edge_impulse_linux.image import ImageImpulseRunner

# --- SETUP HARDWARE ---
# Setup LED
LED_PIN = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)

# Setup LCD (Address biasa 0x27 atau 0x3f)
lcd = CharLCD('PCF8574', 0x27) 
lcd.clear()
lcd.write_string("Sistem Sedia...")

# --- SETUP AI ---
modelfile = "/home/iskandar/kesan_telur.eim"

runner = ImageImpulseRunner(modelfile)
model_info = runner.init()
labels = model_info['model_parameters']['labels']

cap = cv2.VideoCapture(0)

try:
    for res, img in runner.classifier(cap):
        if "bounding_boxes" in res:
            found_something = False
            
            for bb in res["bounding_boxes"]:
                if bb['value'] > 0.7: # Confidence 70%
                    label = bb['label']
                    score = bb['value']
                    found_something = True
                    
                    # Update LCD & LED
                    lcd.clear()
                    lcd.write_string(f"Jumpa: {label}")
                    lcd.cursor_pos = (1, 0)
                    lcd.write_string(f"Conf: {score:.2f}")
                    GPIO.output(LED_PIN, GPIO.HIGH)
                    
                    print(f"Detected: {label}")

            if not found_something:
                GPIO.output(LED_PIN, GPIO.LOW)
                lcd.clear()
                lcd.write_string("Mencari...")

        cv2.imshow('AI Test Hardware', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    runner.stop()

import cv2
import threading
import RPi.GPIO as GPIO
import time
import os
import sys
from edge_impulse_linux.image import ImageImpulseRunner
from RPLCD.i2c import CharLCD

# --- 1. SETUP HARDWARE ---
PUL_PIN = 17 
DIR_PIN = 27 
GPIO.setmode(GPIO.BCM)
GPIO.setup(PUL_PIN, GPIO.OUT)
GPIO.setup(DIR_PIN, GPIO.OUT)

try:
    lcd = CharLCD('PCF8574', 0x27)
    lcd.clear()
    lcd.write_string("Sistem Ready...")
except:
    print("LCD tak dikesan! Check I2C.")

# Global variables
latest_frame = None
latest_result = "Mencari..."
is_running = True

# --- THREAD 1: KAMERA (SMOOTH STREAM) ---
class CameraStream:
    def __init__(self):
        self.stream = cv2.VideoCapture(0)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# --- THREAD 2: LOGIK AI (OBJECT DETECTION) & MOTOR ---
def ai_logic_thread(model_path):
    global latest_result, latest_frame, is_running
    
    with ImageImpulseRunner(model_path) as runner:
        model_info = runner.init()
        print("Berjaya load model AI!")
        
        while is_running:
            if latest_frame is not None:
                # A. Gerak Motor (Sikit-sikit)
                GPIO.output(DIR_PIN, GPIO.HIGH)
                for _ in range(400): 
                    GPIO.output(PUL_PIN, GPIO.HIGH)
                    time.sleep(0.002) 
                    GPIO.output(PUL_PIN, GPIO.LOW)
                    time.sleep(0.002)

                # B. Proses AI (Guna Logic Bounding Boxes)
                features, _ = runner.get_features_from_image(latest_frame)
                res = runner.classify(features)
                
                found_something = False
                
                # Cek hasil bounding boxes
                if "bounding_boxes" in res["result"]:
                    for bb in res["result"]["bounding_boxes"]:
                        label = bb['label']
                        accuracy = bb['value']

                        if accuracy > 0.6: # Set threshold 60%
                            latest_result = f"{label.upper()}"
                            lcd.clear()
                            lcd.write_string(f"Jumpa: {latest_result}\nAcc: {accuracy:.2f}")
                            print(f"Detect: {label} ({accuracy:.2f})")
                            found_something = True
                            break # Ambil satu je yang paling yakin
                
                if not found_something:
                    latest_result = "Scanning..."
                    # lcd.clear()
                    # lcd.write_string("Mencari...")

            time.sleep(0.1)

# --- MAIN DISPLAY THREAD ---
def main():
    global latest_frame, is_running, latest_result
    
    modelfile = "/home/iskandar/telo.eim" # Nama fail kau
    
    if not os.path.exists(modelfile):
        print(f"Error: Fail {modelfile} tak jumpa!")
        return

    cam = CameraStream().start()
    
    t_ai = threading.Thread(target=ai_logic_thread, args=(modelfile,), daemon=True)
    t_ai.start()

    while True:
        frame = cam.read()
        if frame is None: continue
        latest_frame = frame
        
        # Display overlay kat skrin monitor
        cv2.rectangle(frame, (0, 0), (320, 35), (0, 0, 0), -1)
        cv2.putText(frame, f"AI: {latest_result}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('AI Object Detection - Iskandar', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False
            break

    cam.stop()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    lcd.clear()
    lcd.write_string("Sistem Off")

if __name__ == "__main__":
    main()

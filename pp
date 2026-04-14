import cv2
import threading
import RPi.GPIO as GPIO
import time
from edge_impulse_linux.image import ImageImpulseRunner
from RPLCD.i2c import CharLCD # Library untuk LCD I2C

# --- 1. SETUP HARDWARE ---
# Motor NEMA 17
PUL_PIN = 17 
DIR_PIN = 27 
GPIO.setmode(GPIO.BCM)
GPIO.setup(PUL_PIN, GPIO.OUT)
GPIO.setup(DIR_PIN, GPIO.OUT)

# LCD 16x2 I2C (Address 0x27)
try:
    lcd = CharLCD('PCF8574', 0x27)
    lcd.clear()
    lcd.write_string("Sistem Ready...")
except:
    print("LCD tak dikesan! Check wiring SDA/SCL.")

# Global variables
latest_frame = None
latest_result = "Mencari..."
is_running = True

# --- THREAD 1: CAMERA STREAM ---
class CameraStream:
    def __init__(self):
        self.stream = cv2.VideoCapture(0)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
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

# --- THREAD 2: AI & MOTOR LOGIC ---
def ai_logic_thread(model_file):
    global latest_result, latest_frame, is_running
    
    with ImageImpulseRunner(model_file) as runner:
        runner.init()
        while is_running:
            if latest_frame is not None:
                # A. Gerak Motor (Setting 1.2A biar tak panas)
                GPIO.output(DIR_PIN, GPIO.HIGH)
                for _ in range(800): 
                    GPIO.output(PUL_PIN, GPIO.HIGH)
                    time.sleep(0.002) 
                    GPIO.output(PUL_PIN, GPIO.LOW)
                    time.sleep(0.002)

                # B. Proses AI (Fix ValueError)
                features, _ = runner.get_features_from_image(latest_frame)
                res = runner.classify(features)
                
                if "classification" in res:
                    predictions = res['classification']
                    best_label = max(predictions, key=predictions.get)
                    score = predictions[best_label]
                    
                    if score > 0.7:
                        latest_result = f"{best_label.upper()}"
                        # C. Update LCD
                        lcd.clear()
                        lcd.write_string(f"Detected:\r\n{latest_result}")
                    else:
                        latest_result = "Mencari..."
                        lcd.clear()
                        lcd.write_string("Mencari Buah...")

            time.sleep(0.5)

# --- MAIN DISPLAY THREAD ---
def main():
    global latest_frame, is_running
    
    model_path = "/home/iskandar/Aibaru.eim" # Model kau
    cam = CameraStream().start()
    
    t_ai = threading.Thread(target=ai_logic_thread, args=(model_path,), daemon=True)
    t_ai.start()

    print("Sistem Bermula...")

    while True:
        frame = cam.read()
        if frame is None: continue
        latest_frame = frame
        
        # Overlay pada Monitor
        cv2.rectangle(frame, (0, 0), (320, 35), (0, 0, 0), -1)
        cv2.putText(frame, f"Result: {latest_result}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('AI CONVEYOR - ADTEC BP', frame)
        
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

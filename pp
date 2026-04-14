import cv2
import threading
import RPi.GPIO as GPIO
import time
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
    print("LCD tak dikesan!")

latest_frame = None
latest_result = "Mencari..."
is_running = True

# --- THREAD 1: KAMERA ---
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

# --- THREAD 2: AI & MOTOR (DEBUG VERSION) ---
def ai_logic_thread(model_path):
    global latest_result, latest_frame, is_running
    
    with ImageImpulseRunner(model_path) as runner:
        model_info = runner.init()
        # CETAK LABEL YANG ADA DALAM MODEL KAU KAT SHELL
        print(f"Model sedia! Label yang dikesan: {model_info['model_parameters']['labels']}")
        
        while is_running:
            if latest_frame is not None:
                # Gerak motor sikit
                GPIO.output(DIR_PIN, GPIO.HIGH)
                for _ in range(400): 
                    GPIO.output(PUL_PIN, GPIO.HIGH)
                    time.sleep(0.002) 
                    GPIO.output(PUL_PIN, GPIO.LOW)
                    time.sleep(0.002)

                # AI Fikir
                features, _ = runner.get_features_from_image(latest_frame)
                res = runner.classify(features)
                
                if "classification" in res:
                    predictions = res['classification']
                    
                    # --- DEBUG: Cetak semua peratusan kat Shell ---
                    print("\n--- Raw Data AI ---")
                    for label, score in predictions.items():
                        print(f"{label}: {score:.2f}")
                    
                    # Cari yang paling tinggi
                    best_label = max(predictions, key=predictions.get)
                    best_score = predictions[best_label]
                    
                    # KITA TURUNKAN THRESHOLD KE 0.4 BIAR DIA SENSITIF SIKIT
                    if best_score > 0.4:
                        latest_result = f"{best_label.upper()}"
                        lcd.clear()
                        lcd.write_string(f"Item: {latest_result}\nConf: {best_score:.2f}")
                    else:
                        latest_result = "Scanning..."
                        lcd.clear()
                        lcd.write_string("Mencari...")

            time.sleep(0.5)

# --- MAIN DISPLAY ---
def main():
    global latest_frame, is_running, latest_result
    
    model_path = "/home/iskandar/Aibaru.eim"
    cam = CameraStream().start()
    
    t_ai = threading.Thread(target=ai_logic_thread, args=(model_path,), daemon=True)
    t_ai.start()

    while True:
        frame = cam.read()
        if frame is None: continue
        latest_frame = frame
        
        # Display overlay kat Monitor
        cv2.rectangle(frame, (0, 0), (320, 35), (0, 0, 0), -1)
        cv2.putText(frame, f"AI: {latest_result}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('DEBUG AI - ISKANDAR', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False
            break

    cam.stop()
    cv2.destroyAllWindows()
    GPIO.cleanup()

if __name__ == "__main__":
    main()

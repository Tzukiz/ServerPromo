import cv2
import threading
import RPi.GPIO as GPIO
import time
from edge_impulse_linux.image import ImageImpulseRunner
from RPLCD.i2c import CharLCD

# --- 1. KONFIGURASI HARDWARE ---
# Pin Motor NEMA 17
PUL_PIN = 17 
DIR_PIN = 27 
GPIO.setmode(GPIO.BCM)
GPIO.setup(PUL_PIN, GPIO.OUT)
GPIO.setup(DIR_PIN, GPIO.OUT)

# Konfigurasi LCD I2C (Address 0x27)
try:
    lcd = CharLCD('PCF8574', 0x27)
    lcd.clear()
    lcd.write_string("Sistem Ready...")
except:
    print("LCD tidak dikesan. Sila periksa sambungan I2C.")

# Pembolehubah Global
latest_frame = None
latest_result = "Scanning..."
is_running = True

# --- THREAD 1: PENGURUSAN KAMERA (SMOOTH) ---
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

# --- THREAD 2: LOGIK AI & KAWALAN MOTOR ---
def ai_logic_thread(model_path):
    global latest_result, latest_frame, is_running
    
    with ImageImpulseRunner(model_path) as runner:
        runner.init()
        while is_running:
            if latest_frame is not None:
                # Gerakkan konveyor untuk mencari barang
                GPIO.output(DIR_PIN, GPIO.HIGH)
                for _ in range(400): 
                    GPIO.output(PUL_PIN, GPIO.HIGH)
                    time.sleep(0.002) 
                    GPIO.output(PUL_PIN, GPIO.LOW)
                    time.sleep(0.002)

                # Proses pengecaman AI
                features, _ = runner.get_features_from_image(latest_frame)
                res = runner.classify(features)
                
                if "classification" in res:
                    predictions = res['classification']
                    # Cari label dengan nilai keyakinan tertinggi
                    label = max(predictions, key=predictions.get)
                    score = predictions[label]
                    
                    if score > 0.7:
                        # Paparan keputusan pada Shell dan LCD
                        item_found = label.upper()
                        latest_result = f"{item_found} ({score:.2f})"
                        
                        lcd.clear()
                        lcd.write_string(f"Item: {item_found}\r\nConf: {score:.2f}")
                        print(f"Dikesan: {item_found}")
                    else:
                        latest_result = "Scanning..."
                        lcd.clear()
                        lcd.write_string("Mencari barang...")

            time.sleep(0.5) # Rehat seketika untuk CPU

# --- MAIN THREAD: PAPARAN VISUAL ---
def main():
    global latest_frame, is_running
    
    # Path ke fail model anda
    model_path = "/home/iskandar/Aibaru.eim"
    cam = CameraStream().start()
    
    # Mulakan thread AI
    t_ai = threading.Thread(target=ai_logic_thread, args=(model_path,), daemon=True)
    t_ai.start()

    while True:
        frame = cam.read()
        if frame is None: continue
        latest_frame = frame
        
        # Lukis overlay maklumat pada tetingkap kamera
        cv2.rectangle(frame, (0, 0), (320, 40), (0, 0, 0), -1)
        color = (0, 255, 0) if "Scanning" not in latest_result else (0, 255, 255)
        cv2.putText(frame, f"AI: {latest_result}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Fruit Sorting AI - Preview', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False
            break

    cam.stop()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    lcd.clear()
    lcd.write_string("Sistem Tamat")

if __name__ == "__main__":
    main()

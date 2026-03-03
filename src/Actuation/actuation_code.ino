// Use HardwareSerial 1 for the Jetson connection
// Default RX/TX for Serial1 on S3 are often 17/18, but we can define them
#define JETSON_RX 17 
#define JETSON_TX 18

// Pin Definitions
const int RED_PIN = 4;
const int GREEN_PIN = 5;
const int BLUE_PIN = 6;
const int BUZZER_PIN = 9;

void setColor(int r, int g, int b) {
  // analogWrite is the modern, universal way to handle PWM on ESP32
  analogWrite(RED_PIN, r);
  analogWrite(GREEN_PIN, g);
  analogWrite(BLUE_PIN, b);
}

void setup() {
  Serial.begin(115200);
  
  // Initialize Jetson Serial (Using pins defined above)
  // Serial1.begin(115200, SERIAL_8N1, JETSON_RX, JETSON_TX);

  // Define pins as outputs
  pinMode(RED_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  pinMode(BLUE_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  Serial.println("System Initialized: Using analogWrite method.");
}

void loop() {
  if (Serial.available())
  {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input == "NO")
    {
      setColor(255, 0, 0);
    }
    else if (input == "PROCESSING")
    {
      setColor(255, 255, 0);
    }
    else if (input == "COMPLETE")
    {
      setColor(0, 255, 0);
      playsound();
    }
  }
}

void playsound() {
  tone(BUZZER_PIN, 1200, 130);
  delay(160);
  tone(BUZZER_PIN, 2000, 400);
  delay (840);
}
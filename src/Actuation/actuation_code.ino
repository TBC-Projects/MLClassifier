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
  // 1. Not processing
  Serial.println("Color: RED");
  setColor(255, 0, 0);
  delay(1000);

  // 2. Processing
  Serial.println("Color: YELLOW");
  setColor(255, 255, 0);
  delay(1000);

  // 3. Processed
  Serial.println("Color: GREEN");
  setColor(0, 255, 0);
  playsound();
}

void playsound() {
  digitalWrite(BUZZER_PIN, HIGH);
  delay(1000);              // 1 second
  digitalWrite(BUZZER_PIN, LOW);
}

  // Check if the Jetson Nano has sent a new command
  // if (Serial.available() > 0) {
  //   String command = Serial.readStringUntil('\n');
  //   command.trim(); 
  //   command.toLowerCase(); // Handle "Red" or "RED" gracefully

  //   if (command == "red") {
  //     setColor(255, 0, 0);
  //   } else if (command == "green") {
  //     setColor(0, 255, 0);
  //   } else if (command == "blue") {
  //     setColor(0, 0, 255);
  //   } else if (command == "off") {
  //     setColor(0, 0, 0);
  //   } else if (command == "white") {
  //     setColor(255, 255, 255);
  //   }
  //   else if (command == "yellow")
  //   {
  //     setColor (255, 255, 0);
  //   }
  //   // If command is unrecognized, the LED simply stays at its current color
  // }
// }
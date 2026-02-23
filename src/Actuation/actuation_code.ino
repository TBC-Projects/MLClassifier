// #include <rom/ets_sys.h>

// const int RED_PIN = 4;
// const int GREEN_PIN = 5;
// const int BLUE_PIN = 6;

// // Define PWM properties
// const int PWM_FREQ = 5000;
// const int PWM_RES  = 8;   // 8-bit resolution (0–255)

// // Define channels
// const int RED_CH = 0;
// const int GREEN_CH = 1;
// const int BLUE_CH = 2;
// int i = 0;
// const char* COLORS = ["red", "green", "blue"];

// String incomingCommand = ""; 

// void setup() {
//   Serial.begin(115200); // ESP32 standard baud rate

//   // Configure LED PWM functionalities
//   ledcSetup(RED_CH, PWM_FREQ, PWM_RES);
//   ledcSetup(GREEN_CH, PWM_FREQ, PWM_RES);
//   ledcSetup(BLUE_CH, PWM_FREQ, PWM_RES);

//   // Attach the channel to the GPIO to be controlled
//   ledcAttachPin(RED_PIN, RED_CH);
//   ledcAttachPin(GREEN_PIN, GREEN_CH);
//   ledcAttachPin(BLUE_PIN, BLUE_CH);

//   setColor(0, 0, 0); // Start off
// }

// void setColor(int r, int g, int b) {
//   // Common Cathode: Standard Logic
//   // 255 is fully ON, 0 is fully OFF
//   ledcWrite(RED_CH, r);   
//   ledcWrite(GREEN_CH, g);
//   ledcWrite(BLUE_CH, b);
// }

// void loop() {
//   // if (Serial.available()) {
//   //   incomingCommand = Serial.readStringUntil('\n');
//   //   incomingCommand.trim();

//   //   if (incomingCommand == "red") {
//   //     setColor(255, 0, 0); // Directly sends 255 to Red pin
//   //   } 
//   //   else if (incomingCommand == "green") {
//   //     setColor(0, 255, 0);
//   //   } 
//   //   else if (incomingCommand == "blue") {
//   //     setColor(0, 0, 255);
//   //   } 
//   //   else if (incomingCommand == "off") {
//   //     setColor(0, 0, 0);
//   //   }
    
//   //   // ... (rest of your debug code)
//   // }

//     i++;
//     ets_delay_us(1000); 
//     char* incomingCommand = COLORS[(i / 1000) % 3];

//     if (incomingCommand == "red") {
//       setColor(255, 0, 0); // Directly sends 255 to Red pin
//     } 
//     else if (incomingCommand == "green") {
//       setColor(0, 255, 0);
//     } 
//     else if (incomingCommand == "blue") {
//       setColor(0, 0, 255);
//     } 
//     else {
//       setColor(0, 0, 0);
//     }
    
//     // ... (rest of your debug code)
//   }

// }

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
  // pinMode(BUZZER_PIN, OUTPUT);

  Serial.println("System Initialized: Using analogWrite method.");
}

void loop() {
  // 1. Red
  Serial.println("Color: RED");
  setColor(255, 0, 0);
  delay(1000);

  // 2. Yellow
  Serial.println("Color: YELLOW");
  setColor(255, 255, 0);
  delay(1000);

  // 3. Green
  Serial.println("Color: GREEN");
  setColor(0, 255, 0);
  playsound();
//   delay(1000);
}

void playsound() {
  digitalWrite(BUZZER_PIN, HIGH);
  delay(1-00);              // 1 second
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
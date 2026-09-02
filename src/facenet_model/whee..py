import serial

SERIAL_PORT = "/dev/tty.usbmodem1301"  # CHANGE THIS
BAUDRATE = 115200

try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    print(f"Serial port connected")
except Exception as e:
    print(f"Failed connection")
    ser = None

# Send over serial if available
if ser is not None:
    try:
        ser.write(b"COMPLETE\n")
    except Exception as e:
        print(f"Serial write failed")
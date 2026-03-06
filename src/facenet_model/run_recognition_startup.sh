#!/bin/bash

# Facial recognition run on startup

SCRIPT_DIR="src/facenet_model/face_recognition_pipeline.py"
PYTHON_SCRIPT="attendance_logger.py"
LOG_FILE="$SCRIPT_DIR/pipeline.log"

cd $SCRIPT_DIR

echo "=================================" >> $LOG_FILE
echo "Starting pipeline: $(date)" >> $LOG_FILE
echo "=================================" >> $LOG_FILE

while true
do
    echo "Launching recognition pipeline..." | tee -a $LOG_FILE

    # Automatically answer "n" to database rebuild prompt
    printf "n\n" | python3 $PYTHON_SCRIPT >> $LOG_FILE 2>&1

    echo "Pipeline stopped or crashed at $(date)" | tee -a $LOG_FILE
    echo "Restarting in 5 seconds..." | tee -a $LOG_FILE

    sleep 5
done

# RUN THIS AFTER

# chmod +x run_face_pipeline.sh
#./run_face_pipeline.sh

#sudo nano /etc/systemd/system/face_recognition.service

#[Unit]
#Description=Face Recognition Pipeline
#After=network.target

#[Service]
#User=jetson
#WorkingDirectory=/home/jetson/facial_recognition
#ExecStart=/home/jetson/facial_recognition/run_face_pipeline.sh
#Restart=always

#[Install]
#WantedBy=multi-user.target

#sudo systemctl daemon-reload
#sudo systemctl enable face_recognition
#sudo systemctl start face_recognition
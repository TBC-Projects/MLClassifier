# Facial Recognition Attendance Checker

## Introduction
This document is intended to provide a comprehensive overview of our project's developments and serve as a reference tool for new members joining our team.

This report serves two primary functions:

1. **Project Review:** The document provides details on our established objectives, achievements, encountered challenges, key learnings, and prospective future plans. This reflective evaluation allows our team to track progress, make necessary adjustments, and maintain a clear vision moving forward.  
2. **New Member Orientation:** The report also serves as a detailed onboarding guide designed to assist new members in understanding the project's context, objectives, and intricacies. We intend this guide to expedite the acclimatization process and equip newcomers with the necessary knowledge to contribute effectively in the forthcoming semester.

## About the Project

* Project Title:  Facial Recognition Attendance Checker
* Project Purpose: Design and build an embedded system that uses a ML classification algorithm to do real-time image processing. 
* Brief Description:  An embedded system that scans faces in real-time, uses classification to match the face to a person, and marks a person as present. Incorporates user feedback via lights and audio.
* Contributors Involved: Sam Mansouri

## Accomplishments

* Deliverables:  Embedded system device with camera running realtime object tracking and facial recognition with attendance logging.
* Outcome: Successful implementation of device. Pending PCB integration.

## Challenges Faced

* Transmission protocol needed between ESP32 and PC  
  * Solution: Use serial communication 
  * Lesson Learned: design a data pipeline during planning phase  
* Challenge 2 Description: Jetson Nano cannot be imaged
  * Solution: Give to Leonard
  * Lesson Learned: Ensure flash memory (microSD Card) does not have prohibitive technical limitations

## Potential Improvements

* Automated device waking
* PCB integration (delivered but not implemented)
* 3D printed chassis (designed but having printing issues)

## Relevant Resources \[\*\*IMPORTANT\]**

* Key files, photos, figures, algorithms  
* Software/platforms used, and their purpose  
  * Numpy: Mathmatical operations on python
  * Pandas: Large scale data operations on python
  * Jetson Orin Nano: Running real-time object tracking and facial recognition classification model
  * Jetpack 9: Operating system for Jetson platform power computing
* Languages used:  
  * Python
  * C
  * Bash  

## Other Information

* Abbreviations:  
  * ML \- Machine Learning
* Team Roles (People who worked on the project): See ``/doc/HardwareRoles.md`` and ``/doc/SoftwareRoles.md``  
  * Hudson Wong  
    * Software Lead  
  * Adelin Ma  
    * Software Lead  
  * Akshay Ashok  
    * Machine Learning Lead
  * Andrew Bechtel
    * Systems Lead
  * Sam Mansouri
    * Hardware Lead
  * Cliff Pham  
    * Lead Support
  * Arjun Manu
    * Lead Support

# Generating Explanations of Cricket Shot Technqiues

This repository contains all the code that I used the final version of my Research Project

## Creating a Prototype
The code consists of every module that was needed to create the final prototypes seen in the report. To create a new prototype, follow the following instructions:

* Add the video you want to create a prototype for in sample_videos. The videos are split into two groups, ideal_videos (with ideal forms), and practice_videos (for the shots that are to be analyzed).
* In the config.py file add the video name and the target frame to TARGET_FRAME_MAP.
* Make changes to the configuration, to get whatever prototype you want.
* Run main.py to generate the prototypes.

## 📖 Overview

The project aims to:
- Extract key joint angles from cricket pose estimation data.
- Map deviations from an ideal pose to explanation templates.
- Generate personalized, level-specific feedback (e.g., "Bend your front knee slightly more for stability").
- Support both static image feedback and textual feedback using SHAP values.

---

## 🛠️ Key Features

- 🧍 Pose estimation using MediaPipe  
- 📐 Joint angle computation  
- 🧠 SHAP-based expert feedback  
- 🧾 Feedback generation via templates  
- 👩‍🏫 Explanation levels: Beginner, Intermediate, Expert  


## 🤝 Acknowledgements

- MediaPipe for pose estimation  
- SHAP for model interpretability  
- TU Delft BSc Research Project 2025  
````

# Headbanging Detection

This script detects headbanging movements using video input. You can run the script in three modes: evaluating a set of videos, using a webcam, or processing a single video file. Additionally, you can enable plotting for analysis and verbose output for detailed logging.

## Usage
First, ensure you have all the necessary libraries installed, including opencv-python, numpy, and mediapipe.

To run the script, use one of the following commands based on the desired operation mode:

### Webcam Mode
To detect headbanging using your webcam:

```
python main.py --mode webcam [--plot] [--verbose]
```
- --plot: Optional. Enables plotting of variables like angle changes, direction changes, etc.
- --verbose: Optional. Prints messages about direction changes and when headbanging is detected.
### Video Mode
To detect headbanging in a specific video file:

```
python main.py --mode video --video_path <path_to_your_video> [--plot] [--verbose]
```
- --video_path: Required for video mode. Specify the path to the video file.
- --plot and --verbose: Optional flags as explained above.

Evaluation Mode
To evaluate headbanging detection across a folder of video files:

```
python main.py --mode eval
```
This mode automatically processes videos stored in data/headbanging and calculates evaluation metrics.

## Notes
Ensure that the data/headbanging directory exists and contains the video files for evaluation mode.
The script outputs the analysis results, including detected headbanging instances and, if enabled, plots of movement variables.

```
|   config.py
|   download_video.py
|   eval.py
|   headbanging_detection.py
|   main.py
|   plotting_util.py
|   README.md
|   requirements.txt
|   
+---data
|   |   test.csv
|   |   
|   +---headbanging
|   |
|   \---kist_data
```
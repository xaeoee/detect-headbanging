# standard libray in Python for writing user-rriendly command-line interfaces.
import argparse
from headbanging_detection import detect_headbanging
from plotting_util import plot_results
from eval import *

def main():
    parser = argparse.ArgumentParser(description='Headbanging Detection')
    parser.add_argument('--mode', choices=['eval', 'webcam', 'video'], required=True, help='Operation mode: eval, webcam, or video')
    parser.add_argument('--video_path', type=str, default='data/headbanging/video14828.mp4', help='Path to the video file (required for video mode)')
    parser.add_argument('--plot', action='store_true', help='Help to check variables (angle_changes, direction_changes, direction_change_counts)')
    parser.add_argument('--verbose', action='store_true', help='Option to print (direction change, Headbanging Detected)')
    args = parser.parse_args()
    
    if args.mode == 'webcam':
        video_path = 0  # Webcam input
    elif args.mode == 'video':
        if not args.video_path:
            raise ValueError("Video path is required for video mode.")
        video_path = args.video_path
    elif args.mode == 'eval' : 
        video_folder = 'data/headbanging'  # 평가할 비디오가 저장된 폴더 경로    
        results = evaluate_videos(video_folder)
        calculate_metrics(results)

        return 0
    else:
        raise NotImplementedError("Evaluation mode is not implemented in this example.")

    angle_changes, direction_changes, direction_change_counts, x_cord, y_cord, z_cord = detect_headbanging(video_path, args.verbose)
    # if args.plot :
    #     plot_results(angle_changes, direction_changes, direction_change_counts, x_cord, y_cord, z_cord)

if __name__ == "__main__":
    main()

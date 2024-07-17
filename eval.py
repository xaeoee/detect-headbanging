import os
import glob
from headbanging_detection import detect_headbanging
from config import headbanging_threshold
from tqdm import tqdm

def evaluate_videos(video_folder):
    video_files = glob.glob(os.path.join(video_folder, '*.mp4'))
    print(f"Found {len(video_files)} videos for evaluation.")
    
    # 평가 결과 저장용 리스트
    results = []


    print(f"Evaluating videos in folder: {video_folder}")
    
    # Check if the directory exists
    if not os.path.isdir(video_folder):
        print(f"Directory does not exist: {video_folder}")
        return []
    
    # List all files in the directory for debugging
    all_files = os.listdir(video_folder)
    print(f"All files in directory: {all_files}")
    
    # Get all .mp4 files in the directory
    video_files = glob.glob(os.path.join(video_folder, '*.mp4'))
    print(f"Found {len(video_files)} videos for evaluation.")






    for video_path in tqdm(video_files[:5]):
        # print(f"Evaluating {os.path.basename(video_path)}...")
        angle_changes, direction_changes, direction_change_counts, _, _, _ = detect_headbanging(video_path, False)
        
        # "headbanging" 동작 감지 여부 확인
        headbanging_detected = any(count >= headbanging_threshold for count in direction_change_counts)
        results.append(headbanging_detected)

    return results

def calculate_metrics(results):
    true_positives = sum(results)  # "headbanging" 감지된 비디오 수
    false_negatives = len(results) - true_positives  # "headbanging" 누락된 비디오 수
    total = len(results)  # 전체 비디오 수
    
    # 모든 데이터가 True이므로, 정밀도는 계산하지 않음
    recall = true_positives / total if total else 0
    
    # F1 Score 계산 (이 경우 정밀도와 재현율이 동일하므로, F1 Score는 재현율과 같음)
    f1_score = recall

    print(f"Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")
    

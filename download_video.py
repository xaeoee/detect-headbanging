from pytube import YouTube
import pandas as pd
from tqdm import tqdm
import os  # os 모듈 추가

# pip install pytube 

# Kinetics-400 CSV 파일 경로
csv_file_path = 'data/test.csv' 

# CSV 파일에서 데이터 로드
data = pd.read_csv(csv_file_path)

# "headbanging" 라벨을 가진 데이터만 필터링
headbanging_data = data[data['label'] == 'headbanging']

# 다운로드 함수 정의
def download_youtube_videos(video_data, output_path='data/headbanging'):
    for index, row in tqdm(video_data.iterrows(), total=video_data.shape[0]):
        # 비디오 파일 이름을 순서대로 설정
        filename = f'video{index}.mp4'
        file_path = os.path.join(output_path, filename)  # 다운로드될 파일의 전체 경로
        
        # 파일이 이미 존재하는 경우 건너뛰기
        if os.path.exists(file_path):
            # print(f'File {filename} already exists. Skipping...')
            continue
        
        try:
            url = f'https://www.youtube.com/watch?v={row["youtube_id"]}'
            yt = YouTube(url)

            # 가장 높은 해상도의 동영상 다운로드
            yt.streams.get_highest_resolution().download(output_path=output_path, filename=filename)
            # print(f'Video {index} downloaded successfully: {filename}')
        except Exception as e:
            print(f'Error downloading video {index}: {e}')

# 동영상 다운로드 실행
download_youtube_videos(headbanging_data)

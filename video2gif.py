#!/usr/bin/env python
from moviepy.editor import VideoFileClip 
import os 
import argparse

# 비디오 저장 디렉토리 설정 정의
def createDirectory(dir):
    try:
        save_dir = dir + '/gif'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # print(f"SAVE PATH : {os.path.normpath(save_dir)}")
    except OSError:
        print(f"ERROR : 디렉토리 생성하지 못하였음")
    
    return save_dir 

# 비디오 to GIF 파일 저장 함수 정의
def convert_video_to_gif(input_file, output_file):
    clip = VideoFileClip(input_file)
    clip.write_gif(output_file, fps = 30, program = 'ffmpeg')
    
# 인자값 받는 인스턴스 생성
parser = argparse.ArgumentParser(description = 'video to gif')

# 입력받을 인자값 설정 (default 설정 완료)
parser.add_argument('--i', type = str, help = 'input data 경로를 지정하세요.')
parser.add_argument('--o', type = str, default= './', help = f'출력될 파일의 디렉토리를 지정해주세요. ##Default save dir : {os.path.abspath(os.getcwd())}')

# args에 위 내용 저장
args = parser.parse_args()
input_video = args.i

# save directory 저장 
print(f'input video path : {os.path.abspath(input_video)}')
output_gif = os.path.join(createDirectory(args.o),(input_video.split('/')[-1].split('.')[0]+ '.gif'))

print(f'save gif path : {os.path.abspath(output_gif)}')
print('--------Processing file----------')
convert_video_to_gif(input_video, output_gif)
print('------Complete Video to GIF------')


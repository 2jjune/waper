import os
import shutil
import random

# 폴더 경로 설정
source_folder = 'D:/Jiwoon/dataset/window/test/bad/'
destination_folder = 'D:/Jiwoon/dataset/window/train/bad/'

# 'a' 폴더에서 모든 파일 목록 가져오기
files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# 436개의 랜덤 파일 선택
selected_files = random.sample(files, 436)

# 선택된 파일을 'b' 폴더로 이동
for file in selected_files:
    shutil.move(os.path.join(source_folder, file), os.path.join(destination_folder, file))
from PIL import Image
import os

def crop_image_to_9_parts(image_path, save_folder):
    with Image.open(image_path) as img:
        # 이미지 크기를 900x900으로 조정
        img_resized = img.resize((900, 900))

        # 이미지를 9등분하여 저장
        for i in range(3):
            for j in range(3):
                left = i * 300
                top = j * 300
                right = left + 300
                bottom = top + 300

                cropped_img = img_resized.crop((left, top, right, bottom))
                cropped_img.save(os.path.join(save_folder, f'{os.path.basename(image_path).split(".")[0]}_cropped_{i}_{j}.png'))

def process_all_images(folder_path, save_folder):
    # 폴더 내의 모든 이미지 파일에 대해 처리
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            crop_image_to_9_parts(image_path, save_folder)

# 사용 예시
folder_path = 'D:/Jiwoon/dataset/window/test/bad/'
save_folder = 'D:/Jiwoon/dataset/window/300cropped/test/bad/'
process_all_images(folder_path, save_folder)
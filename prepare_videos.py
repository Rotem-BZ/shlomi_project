import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm



def main():
    images_new_size = (64, 64)
    frames_folder = "/datashare/APAS/frames"
    convert_tensor = transforms.PILToTensor()
    for video_folder in os.listdir(frames_folder):
        print("preparing video:", video_folder)
        full_video_path = os.path.join(frames_folder, video_folder)
        frames = sorted(os.listdir(full_video_path))
        frame_vecs = []
        for frame in tqdm(frames):
            frame_path = os.path.join(full_video_path, frame)
            img = Image.open(frame_path)
            img = img.resize(images_new_size)
            frame_vecs.append(convert_tensor(img))
        video_data = torch.tensor(frame_vecs)
        torch.save(video_data, full_video_path + '.pt')



if __name__ == '__main__':
    main()
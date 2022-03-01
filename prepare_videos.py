import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm



def main():
    images_new_size = (64, 64)
    sample_rate = 6
    frames_folder = "/datashare/APAS/frames"
    save_folder = '/home/student/code-rotem/videos_dir/'
    # if os.path.isdir(save_folder):
    #     raise ValueError("save folder already exists")
    # else:
    #     os.mkdir(save_folder)
    convert_tensor = transforms.PILToTensor()
    for video_folder in tqdm(os.listdir(frames_folder)):
        if os.path.isfile(save_folder + video_folder + '.pt'):
            print("already created video:", video_folder)
            continue
        print("preparing video:", video_folder)
        full_video_path = os.path.join(frames_folder, video_folder)
        frames = sorted(os.listdir(full_video_path))
        frame_vecs = []
        i = 0
        for frame in frames:
            i += 1
            if i % sample_rate == 0:
                frame_path = os.path.join(full_video_path, frame)
                img = Image.open(frame_path)
                img = img.resize(images_new_size)
                frame_vecs.append(convert_tensor(img))
        video_data = torch.stack(frame_vecs)
        # print(video_data.shape, "saving to", save_folder +video_folder+ '.pt')
        torch.save(video_data, save_folder + video_folder + '.pt')



if __name__ == '__main__':
    main()
import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt



def main(images_new_size: tuple = (64, 64), save_folder: str = '/home/student/code-rotem/videos_dir/',
         show_image_every: int = -1, sample_rate: int = 6):
    frames_folder = "/datashare/APAS/frames"
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
        print("created folder:", save_folder)
    convert_tensor = transforms.PILToTensor()
    global_frame_index = 0
    for video_folder in tqdm(os.listdir(frames_folder)):
        if os.path.isfile(save_folder + video_folder + '.pt'):
            # print("already created video:", video_folder)
            continue
        # print("preparing video:", video_folder)
        full_video_path = os.path.join(frames_folder, video_folder)
        frames = sorted(os.listdir(full_video_path))
        frame_vecs = []
        i = 0
        for frame in frames:
            i += 1
            if i % sample_rate == 0:
                global_frame_index += 1
                frame_path = os.path.join(full_video_path, frame)
                img = Image.open(frame_path)
                img2 = img.resize(images_new_size)
                frame_vecs.append(convert_tensor(img2))
                if show_image_every != -1 and global_frame_index % show_image_every == 0:
                    fig, (ax1, ax2) = plt.subplots(2)
                    ax1.imshow(img)
                    ax1.set_title("original frame")
                    ax2.imshow(img2)
                    ax2.set_title("resized frame")
                    plt.show()
        video_data = torch.stack(frame_vecs)
        # print(video_data.shape, "saving to", save_folder +video_folder+ '.pt')
        torch.save(video_data, save_folder + video_folder + '.pt')



if __name__ == '__main__':
    main(images_new_size=(224, 224), save_folder='/home/student/code-rotem/videos_dir224/', show_image_every=-1)
    main(images_new_size=(128, 128), save_folder='/home/student/code-rotem/videos_dir128/', show_image_every=-1)
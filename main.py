import os
os.environ["GLOG_minloglevel"] = "2"

# Подавление логов MediaPipe
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from tqdm import tqdm
import multiprocessing
from functools import partial
import glob
import pickle  # Для сохранения результатов

from keypoints import extract_pose_keypoints_timeseries 

def process_video(video_path):
    keypoints = extract_pose_keypoints_timeseries(video_path)
    return (video_path, keypoints)

def main(video_paths, num_workers=4):
    results = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_video, video_paths), total=len(video_paths)):
            results.append(result)
    return results

def get_videos_from_folders(folders, extensions=['*.mp4', '*.avi', '*.mov']):
    videos = []
    for folder in folders:
        for ext in extensions:
            videos.extend(glob.glob(os.path.join(folder, ext)))
    return videos

if __name__ == '__main__':
    folders = [os.path.join('/Users/egor/Documents/STUDY/НИР/FirstImpressionsV2/test', f'test80_{i:02}') for i in range(20)]
    # folders = ['/Users/egor/Documents/STUDY/НИР/FirstImpressionsV2/test/test80_01]', '/Users/egor/Documents/STUDY/НИР/FirstImpressionsV2/test/test80_02']
    all_videos = get_videos_from_folders(folders)
    print(f"Найдено видео: {len(all_videos)}")

    all_results = main(all_videos, num_workers=4)

    # Сохраняем результаты в файл
    output_path = "keypoints_dataset_test.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"Результаты сохранены в {output_path}")

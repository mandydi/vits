import os
from pypinyin import pinyin, Style
import pandas as pd
from sklearn.model_selection import train_test_split
import random

def create_index_and_convert_text(folder_path, output_file):
    with open(output_file, 'w', encoding='utf-8') as output:
        def process_folder(folder, speaker_index):
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        wav_file = file_path.replace('.txt','.wav')
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                            # Convert Chinese text to phonemes using pypinyin
                            pinyin_result = pinyin(text, style=Style.TONE3, heteronym=True)
                            phonemes = ' '.join([item[0] for item in pinyin_result])

                            output_line = f"{wav_file}|{speaker_index}|{phonemes}\n"
                            # output_line = f"{file_path}\t{speaker_index}\t{text}\n"
                            output.write(output_line)
                # Increment speaker index for each subfolder
                speaker_index += 1

                # Recursively process subfolders
                for subfolder in os.listdir(root):
                    subfolder_path = os.path.join(root, subfolder)
                    if os.path.isdir(subfolder_path):
                        speaker_index = process_folder(subfolder_path, speaker_index)
            return speaker_index
        # Start processing from the root folder
        process_folder(folder_path, 1)
        print('已完成数据清洗')

#如果是粤语、重庆歌曲能不能转化




def split_dataset(file_path):
    # 读取数据集文件
    output_train = './filelists/train_lsinger.txt'
    output_val = './filelists/val_lsinger.txt'
    output_test = './filelists/test_lsinger.txt'

    # 读取数据集文件到 pandas DataFrame
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')  # 假设数据集文件是以制表符分隔的

    # 随机打乱数据集
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 划分数据集
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    train_data, temp = train_test_split(df, test_size=(1 - train_ratio), random_state=42)
    val_data, test_data = train_test_split(temp, test_size=test_ratio/(val_ratio + test_ratio), random_state=42)

    # 保存划分后的数据集到新的文件
    train_data.to_csv(output_train, sep='\t', index=False)
    val_data.to_csv(output_val, sep='\t', index=False)
    test_data.to_csv(output_test, sep='\t', index=False)

    print("数据集划分完成，训练集、验证集和测试集已保存到相应文件中。")

if __name__ == "__main__":
    folder_path = 'singer_data'  # 你的文件夹路径
    output_file = 'later_singer_data.txt'  # 所有文件列表
    create_index_and_convert_text(folder_path, output_file)
    split_dataset(output_file)
    print("数据集已完成处理")

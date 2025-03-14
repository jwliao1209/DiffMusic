import os


def count_wav_files(directory):
    return sum(1 for file in os.listdir(directory) if file.lower().endswith('.wav'))


# 指定目錄
directory = "music_data"
wav_count = count_wav_files(directory)
print(f"Number of .wav files in '{directory}': {wav_count}")

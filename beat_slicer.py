import librosa
import subprocess
import numpy as np
import os
from pydub import AudioSegment
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from highlight_detector import HighlightDetector
from file_recognizer import FileRecognizer
from settings import (FIELD_SONG_ID, FIELD_SONGNAME)

def plot_histogram(name: str, save_dir: str, timestamps: List[float], binsize: int, df):
    # Plot the heatmap
    plt.figure(figsize=(15, 3))
    sns.heatmap(df, cmap="YlGnBu", cbar_kws={'label': 'Density'})
    plt.title(f'Timestamp Density for {name}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.savefig(save_dir + 'output.png')
    print("showing plot now:")
    plt.show() 

def analyze_non_lyrical(input_file: str, dir: str) -> Tuple[List[float], Dict[str, List[float]], Dict[str, float], int]:
    slices_dir = dir + "slices"

    # Remove directory if it exists
    subprocess.run(["mkdir", slices_dir.split("/")[0]], check=False)
    subprocess.run(["rm", "-r", slices_dir], check=False)

    # Create directory
    subprocess.run(["mkdir", slices_dir], check=True)

    highlight_detector = HighlightDetector()

    y, sr = librosa.load(input_file, sr=None)

    # Detect beats
    # TODO: develop countermeasure when there are no beat times. 
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    #avg_bpm, beat_start, confidence, tempo, beat_duration = es.RhythmExtractor2013(method='multifeature')(audio)
    song = AudioSegment.from_mp3(input_file) # load song in with pydub

    song_slices = []
    
    # sample 16-beat moments from the song and re-save them as their own slices. 
    i = 0
    while i+16 < len(beat_times):
        song_slice = song[beat_times[i]*1000:beat_times[i+16]*1000] # pydub does things in milliseconds
        song_slices.append(song_slice)
        song_slice.export(f"{slices_dir}/{beat_times[i]}_{beat_times[i+16]}.mp3", format="mp3")
        i += 16
    song_slice = song[beat_times[i]*1000:beat_times[-1]*1000]
    song_slice.export(f"{slices_dir}/{beat_times[i]}_{beat_times[-1]}.mp3", format="mp3")
    
    assert(highlight_detector.get_fingerprinted_songs()==[])

    highlight_detector.fingerprint_directory(f"{slices_dir}", [".mp3"])
    
    '''print("printing songs:")
    highlight_detector.print_df_songs()
    print("printing fingerprints:")
    highlight_detector.print_df_fingerprints()'''

    song_ids = {}
    binsize = 0
    for row in highlight_detector.get_fingerprinted_songs():
        song_id = row[FIELD_SONG_ID]
        song_name = row[FIELD_SONGNAME]
        song_ids[song_name] = song_id
        binsize += 1

    # remove every song from the database one at a time, then re-add into the dictionary.
    # the key re-added into the dictionary should be max_num + 1
    # preprocess the query string such that we can extract the starting point of the best match. 
    # keep an array of the numbers that got returned from the query
    # plot a heatmap of the timestamps that got returned. 

    timestamps = []
    timestamps_graph = {} # used to plot the a graph where each node is a timestamps, to show their relationships

    for file in os.listdir(f"{slices_dir}"): 
        song_name = file.split(".mp3")[0]
        # print(f"Current file: {song_name}")
        try:
            highlight_detector.delete_songs_by_id([song_ids[song_name]]) # delete the file from the db
        except Exception as e:
            if song_name not in song_ids:
                print(f"Song name {song_name} not found in song_id: {song_id}. Parsing must have failed.")
                return 
            print(f"{e}: Failed to delete the following file during moment processing: {song_name} song id: {song_ids[song_name]}")
            return
        results = highlight_detector.recognize(FileRecognizer, f"{slices_dir}/"+file)

        result_names = [result['song_name'] for result in results['results']]
        # print(f"From file we recognized: {result_names}\n")
        timestamps_per_song = []
        for result_name in result_names:
            result_name = result_name.decode().split('_')[0]
            try:   
                timestamps.append(float(result_name))
                timestamps_per_song.append(round(float(result_name), 2))
            except ValueError as e:
                print(f"{e}")
                pass
        # create a dict where the key is the current file name in str
        timestamps_graph[round(float(file.split('_')[0]), 2)] = timestamps_per_song

        try:
            # re-add the song entry into the db
            highlight_detector.fingerprint_file(f"{slices_dir}/{file}") 
        except Exception as e:
            print(f"{e}: Failed to fingerprint the following file: {slices_dir}/{file}")

        song_ids[song_name] = highlight_detector.get_latest_song_id()
        
    with open("results.txt", "w") as f:
        f.write(str(timestamps_graph))
        f.write('\n')

    print(f"\nTimestamps of identified moments: {timestamps}")
    print(f"\nSong moments and their ids: {song_ids}")

    print("\nFile processing was succesfully completed!")
    hist_data, bins = np.histogram(timestamps, bins=binsize) # Adjust the number of bins if needed
    hist_data = hist_data.reshape(-1, 1) # Normalizing the hist_data for the heatmap
    
    time_labels = [(f"{int(bins[i])} - {int(bins[i+1])} s") for i in range(len(bins) - 1)] # Define the time labels for the x-axis
    
    sorted_moments = sorted(song_ids.keys(), key=lambda x: float(x.split("_")[0]))
    max_value = hist_data.argmax()
    timestamp = sorted_moments[max_value]

    df = pd.DataFrame(hist_data, columns=['Density'])
    df.index = time_labels
    df = df.T
    name = input_file
    
    print(f"timestamps: {timestamps}")
    plt.figure(figsize=(15, 3))
    sns.heatmap(df, cmap="YlGnBu", cbar_kws={'label': 'Density'})
    plt.title(f'Timestamp Density for {name}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.savefig(dir + 'output.png')
    print("showing plot now:")
    plt.show() 
    bestchorus_beat_start = float(timestamp.split("_")[0])
    bestchorus_beat_end = float(timestamp.split("_")[1])
        
    # return bestchorus_beat_start, bestchorus_beat_end
    return timestamps, timestamps_graph, song_ids, binsize

if __name__ == '__main__':
    print("Starting file processing...")
    dir = "test7/"
    input_file = "test_files/sdp_interlude.mp3"

    timestamps, timestamps_graph, song_ids, binsize= analyze_non_lyrical(input_file, dir)

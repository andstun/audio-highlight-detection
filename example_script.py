import json
import argparse

from highlight_detector import HighlightDetector
from file_recognizer import FileRecognizer

# load config from a JSON file (or anything outputting a python dictionary)
with open("dejavu.cnf.SAMPLE") as f:
    config = json.load(f)

if __name__ == '__main__':

    # create a HighlightDetector instance
    highlight_detector = HighlightDetector(config)

    # Fingerprint all the mp3's in the directory we give it
    highlight_detector.fingerprint_directory("test", [".wav"])

    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    # Recognize audio from a file
    results = highlight_detector.recognize(FileRecognizer, args.path)
    print(f"From file we recognized: {results}\n")

    # Or recognize audio from your microphone for `secs` seconds
    '''secs = 5
    results = djv.recognize(MicrophoneRecognizer, seconds=secs)
    if results is None:
        print("Nothing recognized -- did you play the song out loud so your mic could hear it? :)")
    else:
        print(f"From mic with {secs} seconds we recognized: {results}\n")'''

    # Or use a recognizer without the shortcut, in anyway you would like
    '''recognizer = FileRecognizer(djv)
    results = recognizer.recognize_file("mp3/Josh-Woodward--I-Want-To-Destroy-Something-Beautiful.mp3")
    print(f"No shortcut, we recognized: {results}\n")'''

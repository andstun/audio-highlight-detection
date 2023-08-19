import multiprocessing
import os
import sys
import traceback
from itertools import groupby
from time import time
from typing import Dict, List, Tuple

import dejavu.logic.decoder as decoder
from dejavu.config.settings import (DEFAULT_FS, DEFAULT_OVERLAP_RATIO,
                                    DEFAULT_WINDOW_SIZE, FIELD_FILE_SHA1, 
                                    FIELD_TOTAL_HASHES, 
                                    FINGERPRINTED_CONFIDENCE, 
                                    FINGERPRINTED_HASHES, HASHES_MATCHED, 
                                    INPUT_CONFIDENCE, INPUT_HASHES, OFFSET, 
                                    OFFSET_SECS, SONG_ID, SONG_NAME, TOPN, 
                                    FIELD_SONG_ID, FIELD_SONGNAME, FIELD_FINGERPRINTED, 
                                    FIELD_FILE_SHA1, FIELD_TOTAL_HASHES, FIELD_HASH,
                                    FIELD_OFFSET)
from dejavu.logic.fingerprint import fingerprint
import pandas as pd

# Create a DataFrame for the SONGS table
SONGS_COLUMNS = {
    f'{FIELD_SONG_ID}': int, # SERIAL in the original schema, represented as int here
    f'{FIELD_SONGNAME}': str,
    f'{FIELD_FINGERPRINTED}': int, # SMALLINT in the original schema
    f'{FIELD_FILE_SHA1}': bytes, # BYTEA in the original schema
    f'{FIELD_TOTAL_HASHES}': int,
}

# Create a DataFrame for the FINGERPRINTS table
FINGERPRINTS_COLUMNS = {
    f'{FIELD_HASH}': bytes, # BYTEA in the original schema
    f'{FIELD_SONG_ID}': int,
    f'{FIELD_OFFSET}': int,
}

class Dejavu:
    def __init__(self):
        # TODO: does a python dict make sense over this?
        self.df_songs = pd.DataFrame(columns=SONGS_COLUMNS.keys()).astype(SONGS_COLUMNS)
        self.df_fingerprints = pd.DataFrame(columns=FINGERPRINTS_COLUMNS.keys()).astype(FINGERPRINTS_COLUMNS)
        self.field_song_id = 1

        # if we should limit seconds fingerprinted,
        # None|-1 means use entire track
        self.limit = None # self.config.get("fingerprint_limit", None)
        if self.limit == -1:  # for JSON compatibility
            self.limit = None
        self.__load_fingerprinted_audio_hashes()

    def __load_fingerprinted_audio_hashes(self) -> None:
        """
        Keeps a dictionary with the hashes of the fingerprinted songs, in that way is possible to check
        whether or not an audio file was already processed.
        """
        # get songs previously indexed
        # self.songs = sself.db.get_songs()
        self.songhashes_set = set()  # to know which ones we've computed before
        for field_file_sha1 in self.df_songs[FIELD_FILE_SHA1]:
            song_hash = field_file_sha1
            self.songhashes_set.add(song_hash)

    def get_latest_song_id(self) -> int:
        return self.field_song_id 
    
    def get_fingerprinted_songs(self) -> List[Dict[str, any]]:
        """
        To pull all fingerprinted songs from the database.

        :return: a list of fingerprinted audios from the database.
        """

        result_df = self.df_songs.loc[
                self.df_songs[FIELD_FINGERPRINTED] == 1,
                [FIELD_SONG_ID, FIELD_SONGNAME]
            ]
        return result_df.to_dict(orient='records')

    def print_df_songs(self, display_max_rows=50) -> None :
        pd.set_option('display.max_rows', display_max_rows)
        print(self.df_songs)

    def print_df_fingerprints(self, display_max_rows=50) -> None :
        pd.set_option('display.max_rows', display_max_rows)
        print(self.df_fingerprints)

    def insert_song(self, song_name, file_hash, hashes) -> None: 
        # insert a new song into the songs dataframe
        new_song = {FIELD_SONG_ID: self.field_song_id, 
                   FIELD_SONGNAME: song_name, 
                   FIELD_FINGERPRINTED: 1,
                   FIELD_FILE_SHA1: bytes.fromhex(file_hash), 
                   FIELD_TOTAL_HASHES: len(hashes)}
        new_song_df = pd.DataFrame(new_song, index=[self.field_song_id])
        self.df_songs = pd.concat([self.df_songs, new_song_df], ignore_index=False)

        # add all of the song's fingerprints to the fingerprints dataframe
        hash_values = [(hsh, self.field_song_id, int(offset)) for hsh, offset in hashes]
        index_values = [self.field_song_id] * len(hash_values)
        hash_values_df = pd.DataFrame(hash_values, columns=FINGERPRINTS_COLUMNS.keys(), index=index_values)
        self.df_fingerprints = pd.concat([self.df_fingerprints, hash_values_df], ignore_index=False)
        self.field_song_id += 1 # increment the field_song_id, which should be unique for every song.

    def insert_song_list(self, songs) -> None: # efficiently add a list of songs in a batch
        return
        new_df_songs = pd.DataFrame(songs, columns=SONGS_COLUMNS.keys(), index=[self.field_song_id]) # TODO: incomplete
        print(new_df_songs)

    def insert_hashes(self, song_id: int, hashes: List[Tuple[str, int]]) -> None:
        return
        values = [(song_id, hsh, int(offset)) for hsh, offset in hashes]
        values_df = pd.DataFrame(values)
        self.df_fingerprints.append(values_df, ignore_index=True)

    def get_song_by_id(self, song_id: int) -> Dict[str, str]:
        result_df = self.df_songs.loc[ # .loc allows both row and column conditions to be specified at once.
            self.df_songs[FIELD_SONG_ID] == song_id,
            [FIELD_SONGNAME, FIELD_FILE_SHA1, FIELD_TOTAL_HASHES]
        ]

        # If you want to apply the upper(encode(..., 'hex')) transformation to the FIELD_FILE_SHA1 column
        result_df[FIELD_FILE_SHA1] = result_df[FIELD_FILE_SHA1].apply(lambda x: x.hex().upper())

        return result_df.to_dict(orient='records')[0]

    def perform_action_on_df() -> None:
        # TODO: implement a general if-else stucture to perform Nonetype checking on df_fingerprints / df_songs
        pass
    
    def delete_songs_by_id(self, song_ids: List[int]) -> None:
        """
        Deletes all audios given their ids.

        :param song_ids: song ids to delete from the database.
        """
        self.df_fingerprints = self.df_fingerprints[~self.df_fingerprints[FIELD_SONG_ID].isin(song_ids)]
        self.df_songs = self.df_songs[~self.df_songs[FIELD_SONG_ID].isin(song_ids)]

    def delete_tables(self): # TODO: handling for the none case needs to be added to INSERT and UPDATE methods
        self.df_fingerprints = None
        self.df_songs = None

    def cleanup_tables(self) -> None: # TODO: none typechecking
        self.df_songs = self.df_songs[self.df_songs[FIELD_FINGERPRINTED] != 0]
        self.df_fingerprints = self.df_fingerprints[~self.df_fingerprints[FIELD_SONG_ID].isin(self.df_songs[FIELD_SONG_ID])]

    def fingerprint_directory(self, path: str, extensions: str, nprocesses: int = None) -> None:
        """
        Given a directory and a set of extensions it fingerprints all files that match each extension specified.

        :param path: path to the directory.
        :param extensions: list of file extensions to consider.
        :param nprocesses: amount of processes to fingerprint the files within the directory.
        """
        # Try to use the maximum amount of processes if not given.
        try:
            nprocesses = nprocesses or multiprocessing.cpu_count()
        except NotImplementedError:
            nprocesses = 1
        else:
            nprocesses = 1 if nprocesses <= 0 else nprocesses

        pool = multiprocessing.Pool(nprocesses)

        filenames_to_fingerprint = []
        for filename, _ in decoder.find_files(path, extensions):
            # don't refingerprint already fingerprinted files
            if decoder.unique_hash(filename) in self.songhashes_set:
                # print(f"{filename} already fingerprinted, continuing...")
                continue

            filenames_to_fingerprint.append(filename)

        # Prepare _fingerprint_worker input
        worker_input = list(zip(filenames_to_fingerprint, [self.limit] * len(filenames_to_fingerprint)))

        # Send off our tasks
        iterator = pool.imap_unordered(Dejavu._fingerprint_worker, worker_input)
        songs = []
        fingerprints = []
        # Loop till we have all of them
        while True:
            try:
                song_name, hashes, file_hash = next(iterator)
            except multiprocessing.TimeoutError:
                continue
            except StopIteration:
                break
            except Exception:
                # print("Failed fingerprinting")
                # Print traceback because we can't reraise it here
                traceback.print_exc(file=sys.stdout)
            else:
                songs.append((self.field_song_id, song_name, 1, bytes.fromhex(file_hash), len(hashes)))
                # don't forget to increment self.field_song_id
                self.insert_song(song_name, file_hash, hashes)

                self.__load_fingerprinted_audio_hashes()

        pool.close()
        pool.join()

    def fingerprint_file(self, file_path: str, song_name: str = None) -> None:
        """
        Given a path to a file the method generates hashes for it and stores them in the database
        for later be queried.

        :param file_path: path to the file.
        :param song_name: song name associated to the audio file.
        """
        song_name_from_path = decoder.get_audio_name_from_path(file_path)
        song_hash = decoder.unique_hash(file_path)
        song_name = song_name or song_name_from_path
        # don't refingerprint already fingerprinted files
        if song_hash in self.songhashes_set:
            pass
            # print(f"{song_name} already fingerprinted, continuing...")
        else:
            song_name, hashes, file_hash = Dejavu._fingerprint_worker(
                (file_path, self.limit),
                song_name=song_name
            )
            self.insert_song(song_name, file_hash, hashes)

            self.__load_fingerprinted_audio_hashes()

    def generate_fingerprints(self, samples: List[int], Fs=DEFAULT_FS) -> Tuple[List[Tuple[str, int]], float]:
        f"""
        Generate the fingerprints for the given sample data (channel).

        :param samples: list of ints which represents the channel info of the given audio file.
        :param Fs: sampling rate which defaults to {DEFAULT_FS}.
        :return: a list of tuples for hash and its corresponding offset, together with the generation time.
        """
        t = time()
        hashes = fingerprint(samples, Fs=Fs)
        fingerprint_time = time() - t
        return hashes, fingerprint_time
    
    def return_matches(self, hashes: List[Tuple[str, int]],
                       batch_size: int = 1000) -> Tuple[List[Tuple[int, int]], Dict[int, int]]:
        """
        Searches the database for pairs of (hash, offset) values.

        :param hashes: A sequence of tuples in the format (hash, offset)
            - hash: Part of a sha1 hash, in hexadecimal format
            - offset: Offset this hash was created from/at.
        :param batch_size: number of query's batches.
        :return: a list of (sid, offset_difference) tuples and a
        dictionary with the amount of hashes matched (not considering
        duplicated hashes) in each song.
            - song id: Song identifier
            - offset_difference: (database_offset - sampled_offset)
        """
        # Create a dictionary of hash => offset pairs for later lookups

        mapper = {}
        for hsh, offset in hashes:
            if hsh.lower() in mapper.keys():
                mapper[hsh.lower()].append(offset)
            else:
                mapper[hsh.lower()] = [offset]

        values_hex = list(mapper.keys())
        values_hex_lower = [value.lower() for value in values_hex]

        # in order to count each hash only once per db offset we use the dic below
        dedup_hashes = {}
        results = []

        result_df = self.df_fingerprints[self.df_fingerprints[FIELD_HASH].isin(values_hex_lower)]

        for hsh, sid, offset in result_df.itertuples(index=False): # iterrows is slow, numpy vectorization is fastest
            if sid not in dedup_hashes.keys():
                dedup_hashes[sid] = 1
            else:
                dedup_hashes[sid] += 1
            #  we now evaluate all offset for each  hash matched
            for song_sampled_offset in mapper[hsh]:
                results.append((sid, offset - song_sampled_offset))

        return results, dedup_hashes

    def find_matches(self, hashes: List[Tuple[str, int]]) -> Tuple[List[Tuple[int, int]], Dict[str, int], float]:
        """
        Finds the corresponding matches on the fingerprinted audios for the given hashes.

        :param hashes: list of tuples for hashes and their corresponding offsets
        :return: a tuple containing the matches found against the db, a dictionary which counts the different
         hashes matched for each song (with the song id as key), and the time that the query took.

        """
        t = time()
        matches, dedup_hashes = self.return_matches(hashes)
        query_time = time() - t

        return matches, dedup_hashes, query_time

    def align_matches(self, matches: List[Tuple[int, int]], dedup_hashes: Dict[str, int], queried_hashes: int,
                      topn: int = TOPN) -> List[Dict[str, any]]:
        """
        Finds hash matches that align in time with other matches and finds
        consensus about which hashes are "true" signal from the audio.

        :param matches: matches from the database
        :param dedup_hashes: dictionary containing the hashes matched without duplicates for each song
        (key is the song id).
        :param queried_hashes: amount of hashes sent for matching against the db
        :param topn: number of results being returned back.
        :return: a list of dictionaries (based on topn) with match information.
        """
        # count offset occurrences per song and keep only the maximum ones.
        sorted_matches = sorted(matches, key=lambda m: (m[0], m[1]))
        counts = [(*key, len(list(group))) for key, group in groupby(sorted_matches, key=lambda m: (m[0], m[1]))]
        songs_matches = sorted(
            [max(list(group), key=lambda g: g[2]) for key, group in groupby(counts, key=lambda count: count[0])],
            key=lambda count: count[2], reverse=True
        )

        songs_result = []
        for song_id, offset, _ in songs_matches[0:topn]:  # consider topn elements in the result
            song = self.get_song_by_id(song_id)
            song_name = song.get(SONG_NAME, None)
            song_hashes = song.get(FIELD_TOTAL_HASHES, None)
            nseconds = round(float(offset) / DEFAULT_FS * DEFAULT_WINDOW_SIZE * DEFAULT_OVERLAP_RATIO, 5)
            hashes_matched = dedup_hashes[song_id]

            song = {
                SONG_ID: song_id,
                SONG_NAME: song_name.encode("utf8"),
                INPUT_HASHES: queried_hashes,
                FINGERPRINTED_HASHES: song_hashes,
                HASHES_MATCHED: hashes_matched,
                # Percentage regarding hashes matched vs hashes from the input.
                INPUT_CONFIDENCE: round(hashes_matched / queried_hashes, 2),
                # Percentage regarding hashes matched vs hashes fingerprinted in the db.
                FINGERPRINTED_CONFIDENCE: round(hashes_matched / song_hashes, 2),
                OFFSET: offset,
                OFFSET_SECS: nseconds,
                FIELD_FILE_SHA1: song.get(FIELD_FILE_SHA1, None).encode("utf8")
            }

            songs_result.append(song)

        return songs_result

    def recognize(self, recognizer, *options, **kwoptions) -> Dict[str, any]:
        r = recognizer(self)
        return r.recognize(*options, **kwoptions)

    @staticmethod
    def _fingerprint_worker(arguments, song_name=None):
        # Pool.imap sends arguments as tuples so we have to unpack
        # them ourself.
        try:
            file_name, limit = arguments
        except ValueError:
            pass

        song_name, extension = os.path.splitext(os.path.basename(file_name))

        fingerprints, file_hash = Dejavu.get_file_fingerprints(file_name, limit, print_output=True)

        return song_name, fingerprints, file_hash

    @staticmethod
    def get_file_fingerprints(file_name: str, limit: int, print_output: bool = False):
        channels, fs, file_hash = decoder.read(file_name, limit)
        fingerprints = set()
        channel_amount = len(channels)
        for channeln, channel in enumerate(channels, start=1):
            if print_output:
                pass
                #print(f"Fingerprinting channel {channeln}/{channel_amount} for {file_name}")

            hashes = fingerprint(channel, Fs=fs)

            if print_output:
                pass
                #print(f"Finished channel {channeln}/{channel_amount} for {file_name}")

            fingerprints |= set(hashes)

        return fingerprints, file_hash

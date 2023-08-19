import abc
from time import time
from typing import Dict, List, Tuple

import numpy as np

from settings import DEFAULT_FS


class BaseRecognizer(object, metaclass=abc.ABCMeta):
    def __init__(self, highlight_detector):
        self.highlight_detector = highlight_detector
        self.Fs = DEFAULT_FS

    def _recognize(self, *data) -> Tuple[List[Dict[str, any]], int, int, int]:
        fingerprint_times = []
        hashes = set()  # to remove possible duplicated fingerprints we built a set.
        for channel in data:
            fingerprints, fingerprint_time = self.highlight_detector.generate_fingerprints(channel, Fs=self.Fs)
            fingerprint_times.append(fingerprint_time)
            hashes |= set(fingerprints)

        matches, dedup_hashes, query_time = self.highlight_detector.find_matches(hashes)
        #print(f"Matches: {matches}")
        t = time()
        final_results = self.highlight_detector.align_matches(matches, dedup_hashes, len(hashes))
        align_time = time() - t

        return final_results, np.sum(fingerprint_times), query_time, align_time

    @abc.abstractmethod
    def recognize(self) -> Dict[str, any]:
        pass  # base class does nothing

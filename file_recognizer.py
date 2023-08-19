from time import time
from typing import Dict

import decoder
from base_recognizer import BaseRecognizer
from settings import (ALIGN_TIME, FINGERPRINT_TIME, QUERY_TIME,
                                    RESULTS, TOTAL_TIME)


class FileRecognizer(BaseRecognizer):
    def __init__(self, highlight_detector):
        super().__init__(highlight_detector)

    def recognize_file(self, filename: str) -> Dict[str, any]:
        channels, self.Fs, _ = decoder.read(filename, self.highlight_detector.limit)

        t = time()
        matches, fingerprint_time, query_time, align_time = self._recognize(*channels)
        t = time() - t

        results = {
            TOTAL_TIME: t,
            FINGERPRINT_TIME: fingerprint_time,
            QUERY_TIME: query_time,
            ALIGN_TIME: align_time,
            RESULTS: matches
        }

        return results

    def recognize(self, filename: str) -> Dict[str, any]:
        return self.recognize_file(filename)

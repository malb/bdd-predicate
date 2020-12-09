# -*- coding: utf-8 -*-
"""
"""

import os
import sys
import io


def btoi(b):
    return int.from_bytes(b, "big")


def itob(i, baselen):
    return int.to_bytes(int(i), length=baselen, byteorder="big")


class SuppressStream(object):
    """
    Suppress errors (being printed by FPLLL, which are to be expected).
    """

    def __init__(self, stream=sys.stderr):
        try:
            self.orig_stream_fileno = stream.fileno()
            self.skip = False
        except io.UnsupportedOperation:
            self.skip = True

    def __enter__(self):
        if self.skip:
            return
        self.orig_stream_dup = os.dup(self.orig_stream_fileno)
        self.devnull = open(os.devnull, "w")
        os.dup2(self.devnull.fileno(), self.orig_stream_fileno)

    def __exit__(self, type, value, traceback):
        if self.skip:
            return
        os.close(self.orig_stream_fileno)
        os.dup2(self.orig_stream_dup, self.orig_stream_fileno)
        os.close(self.orig_stream_dup)
        self.devnull.close()

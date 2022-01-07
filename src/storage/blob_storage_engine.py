from src.catalog.models.df_metadata import DataFrameMetadata
from src.storage.petastorm_storage_engine import AbstractStorageEngine
from src.models.storage.batch import Batch
from typing import Iterator
import os
import ffmpeg
import numpy as np
import pandas as pd
from PIL import Image


class BlobStorageEngine(AbstractStorageEngine):

    """
    Thread-unsafe storage engine that writes an entire sequence of frames out
    as a video, sequentially A video is a _baked_ format TODO: allow access
    to multiple tables at the same time (not multithreaded access to the
    same table)
    - This can be achieved using a dictionary mapping from table name to the
    corresponding process
    """

    STORAGE_PATH = "blob_data"

    def __init__(self):
        self.frames = None

    def _url(self, table: DataFrameMetadata):
        return "/".join([self.STORAGE_PATH, table.file_url])

    def get_frame_count(self, table):
        directory_url = self._url(table)
        frame_file = os.path.join(directory_url, "frames.txt")
        with open(frame_file, "r") as fp:
            return int(fp.readline().rstrip())

    def create(self, table: DataFrameMetadata):
        if not os.path.exists(self.STORAGE_PATH):
            os.makedirs(self.STORAGE_PATH)
        # assert: directory present for storing files
        folder_location = self._url(table)

        # create empty directory in filesystem
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)

    def open_write(self, table: DataFrameMetadata):
        self.frame_id = 0
        pass

    def write(self, table: DataFrameMetadata, rows: Batch):
        height, width = table.resolution
        frames = rows.frames['data'].values

        folder_location = self._url(table)

        for frame in frames:
            image = Image.fromarray(frame)
            image.save("/".join([folder_location, "{}.jpeg".format(self.frame_id)]))

            self.frame_id += 1

    def open_read(self, table: DataFrameMetadata, predicate=None):

        assert self.frames == None

        folder_location = self._url(table)

        if predicate is None:
            frame_count_file = "/".join([folder_location, "frames.txt"])
            frame_no = 0
            with open(frame_count_file, "r") as fp:
                frames_end = int(fp.readline().rstrip())

            self.frames = iter(range(frame_no, frames_end))


        else:
            pred_type, pred_value = predicate
            filter_expr = ""
            if pred_type == "between":
                frame_no, frames_end = pred_value
                self.frames = iter(range(frame_no, frames_end))
            elif pred_type == "random":
                self.frames = iter(sorted(pred_value))
            else:
                raise ValueError

    def read(self, table, batch_mem_size=30000000) -> Iterator[Batch]:
        """
        table: DataframeMetadata
        pos: frame number
        select-expr: ffmpeg select expression
        """

        assert self.frames != None

        data = []

        while True:
            try: 
                next_frame_no = next(self.frames)
            except StopIteration:
                break

            frame_file = "/".join([self._url(table),
                "{}.jpeg".format(next_frame_no)])

            in_frame = Image.open(frame_file)
            in_array = np.asarray(in_frame)

            data.append({'id': next_frame_no, 'data': in_array})
            width, height, rgb = in_array.shape
            row_size = width * height * rgb
            if len(data) * row_size > batch_mem_size:
                yield Batch(pd.DataFrame(data))
                data = []

        if len(data) > 0:
            yield Batch(pd.DataFrame(data))

    def _open(self, table):
        pass

    def _close(self, table):
        pass

    def close_write(self, table):
        # use this to stop the async ffmpeg process
        with open(os.path.join(self._url(table), "frames.txt"), "w") as fp:
            print(self.frame_id, file = fp)
        self.frame_id = None


    def close_read(self, table):
        self.frames = None

    def _read_init(self, table):
        # use this to start the async ffmpeg read process
        pass



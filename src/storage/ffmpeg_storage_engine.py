from src.catalog.models.df_metadata import DataFrameMetadata
from src.storage.petastorm_storage_engine import AbstractStorageEngine
from src.models.storage.batch import Batch
from typing import Iterator
import os
import ffmpeg
import numpy as np
import pandas as pd


class FFmpegStorageEngine(AbstractStorageEngine):

    """
    Thread-unsafe storage engine that writes an entire sequence of frames out
    as a video, sequentially A video is a _baked_ format TODO: allow access
    to multiple tables at the same time (not multithreaded access to the
    same table)
    - This can be achieved using a dictionary mapping from table name to the
    corresponding process
    """

    STORAGE_PATH = "ffmpeg_data"
    STORAGE_FORMAT = "mp4"

    def __init__(self):
        self.location = None
        self.process = None

    def _url(self, table: DataFrameMetadata):
        return "/".join([self.STORAGE_PATH, table.file_url]) + "." + self.STORAGE_FORMAT

    def get_frame_count(self, table: DataFrameMetadata):
        file_url = self._url(table)
        probe = ffmpeg.probe(file_url)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        n_frames = int(video_stream['nb_frames'])
        return n_frames

    def create(self, table: DataFrameMetadata):
        if not os.path.exists(self.STORAGE_PATH):
            os.makedirs(self.STORAGE_PATH)
        # assert: directory present for storing files
        file_location = self._url(table)

        # create empty file in filesystem
        open(file_location, "w").close()

    def open_write(self, table: DataFrameMetadata):
        assert self.location is None and self.process is None, (
            self.location, self.process)
        file_location = self._url(table)
        width, height = table.resolution
        self.process = (
            ffmpeg
            .input('pipe:', format='rawvideo',
                   pix_fmt='bgr24', s='{}x{}'.format(width, height))
            .output(file_location,
                    format=self.STORAGE_FORMAT, crf='18')
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
        )

    def write(self, table: DataFrameMetadata, rows: Batch):
        height, width = table.resolution
        frames = rows.frames['data'].values

        for frame in frames:

            self.process.stdin.write(
                frame
                .astype(np.uint8)
                .tobytes()
            )

    def open_read(self, table: DataFrameMetadata, predicate=None):
        assert self.process is None and self.location is None
        file_location = self._url(table)
        width, height = get_resolution(file_location)
        table.resolution = (width, height)

        if predicate is None:

            self.process = (
                ffmpeg
                .input(file_location)
                .output('pipe:', format='rawvideo',
                        pix_fmt='rgb24', s='{}x{}'.format(width, height))
                .run_async(pipe_stdout=True, quiet=True)
            )

        else:
            pred_type, pred_value = predicate
            filter_expr = ""
            if pred_type == "between":
                start, end = pred_value
                filter_expr = "between(n,{},{})".format(start, end)
            elif pred_type == "random":
                filter_expr = "+".join(["eq(n,{})".format(x)
                                        for x in sorted(pred_value)])
            else:
                raise ValueError

            stream = (
                ffmpeg
                .input(file_location)
                .filter('select', filter_expr)
                .output('pipe:', format='rawvideo', vsync='0',
                        pix_fmt='rgb24', s='{}x{}'.format(width, height))
            )

            #print(stream.compile())
            self.process = stream.run_async(pipe_stdout=True, quiet=True)

    def read(self, table, batch_mem_size=30000000) -> Iterator[Batch]:
        """
        table: DataframeMetadata
        pos: frame number
        select-expr: ffmpeg select expression
        """

        i = 0
        data = []
        width, height = table.resolution
        row_size = width * height * 3

        while True:
            in_bytes = self.process.stdout.read(width * height * 3)

            if not in_bytes:
                break

            in_frame = (
                np
                .frombuffer(in_bytes, np.uint8)
                .reshape([height, width, 3])
            )

            data.append({'id': i, 'data': in_frame})

            if len(data) * row_size > batch_mem_size:
                yield Batch(pd.DataFrame(data))
                data = []

            i += 1

        if len(data) > 0:
            yield Batch(pd.DataFrame(data))

    def _open(self, table):
        pass

    def _close(self, table):
        pass

    def close_write(self, table):
        # use this to stop the async ffmpeg process
        self.process.stdin.close()
        self.process.wait()
        self.process = None
        self.location = None

    def close_read(self, table):
        self.process.wait()
        self.process = None
        self.location = None

    def _read_init(self, table):
        # use this to start the async ffmpeg read process
        pass


def get_resolution(file_url):
    probe = ffmpeg.probe(file_url)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    return (width, height)

def get_frame_count(file_url):
    probe = ffmpeg.probe(file_url)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    n_frames = int(video_stream['nb_frames'])
    return n_frames

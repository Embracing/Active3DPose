import logging
import os
import time
from multiprocessing import Process, Queue
from pathlib import Path

import av
import numpy as np
import zarr


def create_logger(cfg, video_basename):
    final_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not final_output_dir.exists():
        print(f'=> Output dir does not exist. Creating {final_output_dir}')
        final_output_dir.mkdir()

    # final_output_dir = root_output_dir
    # print('=> creating {}'.format(final_output_dir))
    # final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    log_file = f'{time_str}_{video_basename}.log'

    log_dir = Path(cfg.LOG_DIR)
    if not log_dir.exists():
        print(f'=> Log dir does not exist. Creating {log_dir}')
        log_dir.mkdir()

    final_log_file = log_dir / log_file
    print(f'=> Creating {final_log_file}')

    head = '%(asctime)-15s-%(name)s-%(levelname)s: %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger, str(final_output_dir), time_str


def read_anim_names(path):
    names = []
    with open(path) as f:
        lines = f.readlines()
        names = list(map(lambda x: x.strip(), lines))
    return names


def save_render_zarray(save_dict, path):
    group = zarr.open(path, mode='a')
    for k, v in save_dict.items():
        if k in group:
            group[k].append(v)
        else:
            chunks_shape = (10000,) + tuple(None for _ in range(v.ndim - 1))
            group.array(k, v, chunks=chunks_shape)


class ZarrBufferSaver:
    def __init__(self, shape_dict, save_path, capacity=100000):
        assert capacity >= 1
        self.buffer_dict = dict()
        for k, v in shape_dict.items():
            self.buffer_dict[k] = np.empty((capacity,) + v, dtype=np.float32)

        self.pointer = 0
        self.capacity = capacity
        self.save_path = save_path

    def append(self, data_dict):
        """
        add one data item
        """
        if self.pointer >= self.capacity:
            self.save()

        for k, v in data_dict.items():
            self.buffer_dict[k][self.pointer] = v
        self.pointer += 1

    def save(self):
        if self.pointer == 0:
            return
        group = zarr.open(self.save_path, mode='a')
        for k, v in self.buffer_dict.items():
            if k in group:
                group[k].append(v[: self.pointer])
            else:
                chunks_shape = (10000,) + tuple(None for _ in range(v.ndim - 1))
                group.array(k, v[: self.pointer], chunks=chunks_shape)

        self.clear()

    def clear(self):
        self.pointer = 0

    def close(self):
        """
        Save rest data in the buffer.
        """
        if self.pointer > 0:
            self.save()

    def __len__(self):
        return self.pointer


class ZarrVariableBufferSaver:
    def __init__(self, shape_dict, save_path, var_maxlen=10, capacity=100000, map='Blank'):
        assert capacity >= 1

        # for fixed shape array
        # gt3d, corners_world_list,
        # pred3d, variable len, shape=None
        self.buffer_dict = dict()
        for k, v in shape_dict.items():
            if v is not None:
                self.buffer_dict[k] = np.empty((capacity,) + v, dtype=np.float32)
            else:
                self.buffer_dict[k] = zarr.empty(
                    (capacity, var_maxlen), dtype='array:f4', chunks=(10000, None)
                )

        self.pointer = 0
        self.capacity = capacity
        self.save_path = save_path
        self.var_maxlen = var_maxlen

        # save meta info
        store = zarr.DirectoryStore(self.save_path)
        group = zarr.open_group(store=store, mode='a')
        group.attrs['map'] = map

    def append(self, data_dict):
        """
        add one data item
        """
        if self.pointer >= self.capacity:
            self.save()

        for k, v in data_dict.items():
            # for varLen array, v : zarr ragged array
            # for fixedLen array, v: numpy array
            self.buffer_dict[k][self.pointer] = v
        self.pointer += 1

    def save(self):
        if self.pointer == 0:
            return
        store = zarr.DirectoryStore(self.save_path)
        group = zarr.open_group(store=store, mode='a')
        # group = zarr.open(self.save_path, mode='a')
        for k, v in self.buffer_dict.items():
            if k in group:
                group[k].append(v[: self.pointer])
            else:
                # create array in group
                if isinstance(v, zarr.core.Array):
                    group.array(k, v[: self.pointer], dtype='array:f4')
                else:
                    chunks_shape = (10000,) + tuple(None for _ in range(v.ndim - 1))
                    group.array(k, v[: self.pointer], chunks=chunks_shape)
        self.clear()

    def clear(self):
        self.pointer = 0

    def close(self):
        """
        Save rest data in the buffer.
        """
        if self.pointer > 0:
            self.save()

    def __len__(self):
        return self.pointer


class VideoSaver:
    def __init__(self, num_cam, save_dir):
        """
        Save video from different views to different video files
        under save_dir (timestap), cam1.mp4, cam2.mp4, cam3.mp4

        Pipeline: init() --> append() --> close()
        """
        # assert capacity >= 1
        assert num_cam >= 1

        self.num_cam = num_cam
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def init(self):
        self.queue_list = [Queue() for _ in range(self.num_cam)]
        self.process_list = [
            Process(
                target=self.encode_and_write,
                args=(self.queue_list[idx], self.save_dir, idx),
            )
            for idx in range(self.num_cam)
        ]
        for p in self.process_list:
            p.start()

    def __getstate__(self):
        state = self.__dict__.copy()

        # don't pickle the process, it is unpicklable.
        del state['process_list']
        del state['queue_list']
        del state['num_cam']
        del state['save_dir']
        return state

    def append(self, data_list):
        """
        Add one data item
        Input: frames from C views at a single timestap. [C], [h, w, 3]
        """
        assert len(data_list) == self.num_cam, 'input len should be %d' % self.num_cam

        for v, q in zip(data_list, self.queue_list):
            q.put(v)

    def close(self):
        print('=> waiting for video saver process to finish')
        for q in self.queue_list:
            q.put(None)

        for idx, p in enumerate(self.process_list):
            count = 0
            while p.is_alive():  # automatic join dead process
                print(
                    '=> Info: Process %d is alive, waiting to close, %d sec' % (idx, count),
                    end='\r',
                    flush=True,
                )
                count += 1
                time.sleep(1)
            else:
                p.join()  # redundant, but clear!

        print('\n=> Video saver finished !')

    def encode_and_write(self, q, save_dir, idx):
        with av.open(os.path.join(save_dir, 'view_%d.mp4' % idx), 'w') as container:
            stream = container.add_stream('libx264', '60')
            stream.pix_fmt = 'yuv420p'

            # stream.width = 320
            # stream.height = 240

            stream.options['crf'] = '15'  # for libx264, 0-51, 0 is lossless.

            # encoding
            while True:
                img = q.get()  # block for items, exit when 'None' is got,
                if img is None:
                    break
                frame = av.VideoFrame.from_ndarray(img)
                for packet in stream.encode(frame):
                    container.mux(packet)

            # flush
            for packet in stream.encode():
                container.mux(packet)

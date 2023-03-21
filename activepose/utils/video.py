import logging
import shlex
import subprocess as sp

import av
import numpy as np


logger = logging.getLogger(__name__)


def get_resolution(filename):
    # Resolution before rotation metadata is applied
    command = [
        'ffprobe',
        '-v',
        'error',
        '-select_streams',
        'v:0',
        '-show_entries',
        'stream=width,height',
        '-of',
        'csv=p=0',
        filename,
    ]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            w, h = line.decode().strip().split(',')
            return int(w), int(h)


def get_fps(filename):
    command = [
        'ffprobe',
        '-v',
        'error',
        '-select_streams',
        'v:0',
        '-show_entries',
        'stream=r_frame_rate',
        '-of',
        'csv=p=0',
        filename,
    ]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            a, b = line.decode().strip().split('/')
            return int(a) / int(b)


def get_rotation(filename):
    command = shlex.split(
        'ffprobe -loglevel error -select_streams v:0 \
     -show_entries stream_tags=rotate -of default=nw=1:nk=1 %s'
        % filename
    )
    process = sp.run(command, stdout=sp.PIPE, universal_newlines=True)
    angle = process.stdout.strip('\n')
    if angle != '':
        angle = int(angle)
    else:
        angle = 0
    return angle


def get_duration(filename):
    """
    Container duration
    """
    command = shlex.split(
        'ffprobe -loglevel error -select_streams v:0 \
     -show_entries stream=duration -of default=nw=1:nk=1 %s'
        % filename
    )
    process = sp.run(command, stdout=sp.PIPE, universal_newlines=True)
    duration = process.stdout.strip('\n')
    return float(duration)


def get_metadata(filename):
    """
    Return:
        Dictionary {tag:value}. tag:None for empty value
    """
    tags = 'width,height,r_frame_rate,avg_frame_rate,duration'
    command = shlex.split(
        'ffprobe -loglevel error -select_streams v:0 \
     -show_entries stream=%s -of default=nw=1 %s'
        % (tags, filename)
    )
    process = sp.run(command, stdout=sp.PIPE, universal_newlines=True)
    output = process.stdout.splitlines()
    metadata = {}
    for item in output:
        tag, value = item.split('=')
        if value == 'N/A':
            metadata[tag] = np.nan
            continue
        else:
            if tag in ['width', 'height']:
                metadata[tag] = int(value)
            elif tag in ['duration']:
                metadata[tag] = float(value)
            elif tag in ['r_frame_rate', 'avg_frame_rate']:
                a, b = value.split('/')
                metadata[tag] = int(a) / int(b)
            else:
                assert 0, 'not defined metadata %s' % tag
    return metadata


def rotate_frame(frames, angle):
    """
    frames: [N, h, w, c] or [h, w, c]
    rotate couter clock-wise
    """
    if angle == 0:
        return frames

    if frames.ndim == 4:
        plane = (1, 2)
    else:
        plane = (0, 1)

    if angle == 90:
        frames = np.rot90(frames, k=-1, axes=plane)
    elif angle == 180:
        frames = np.rot90(frames, k=-2, axes=plane)
    elif angle == 270:
        frames = np.rot90(frames, k=1, axes=plane)
    return frames


def read_video(filename, skip=0, limit=-1):
    w, h = get_resolution(filename)

    command = [
        'ffmpeg',
        '-i',
        filename,
        '-f',
        'image2pipe',
        '-pix_fmt',
        'rgb24',
        '-vsync',
        '0',
        '-vcodec',
        'rawvideo',
        '-',
    ]

    i = 0
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        while True:
            data = pipe.stdout.read(w * h * 3)
            if not data:
                break
            i += 1
            if i > limit and limit != -1:
                continue
            if i > skip:
                yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))


def read_video_pyav(filename, bgr=True):
    imgs = []
    with av.open(filename, metadata_errors='ignore') as container:
        for frame in container.decode(video=0):
            rgb_img = np.asarray(frame.to_image())
            imgs.append(rgb_img)
    imgs = np.array(imgs)  # [N, h, w, c]

    # get rotate metadata
    angle = get_rotation(filename)
    imgs = rotate_frame(imgs, angle=angle)

    logger.info('=> Metadata rotation: %d' % angle)

    if bgr:
        imgs = imgs[:, :, :, ::-1]  # RGB to BGR

    imgs = np.ascontiguousarray(imgs)
    return imgs

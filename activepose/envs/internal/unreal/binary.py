# Build script of unrealcv, supports win, linux and mac.
# A single file library
# Weichao Qiu @ 2017

import abc
import atexit
import os
import platform
import subprocess
import time


def get_platform_name():
    """'
    Python and UE4 use different names for platform, in this script we will use UE4 platform name exclusively
    """
    py2UE4 = {
        # pyname : ue4name
        'Darwin': 'Mac',
        'Windows': 'Win64',
        'Linux': 'Linux',
    }
    # Key: python platform name, Value: UE4
    platform_name = py2UE4.get(platform.system())
    if not platform_name:
        print('Can not recognize platform %s' % platform.system())
    return platform_name


def UE4Binary(binary_path):
    """
    Return a platform-dependent binary for user.
    Examples
    --------
    >>> binary = UE4Binary('./WindowsNoEditor/RealisticRendering.exe')  # For windows
    >>> binary = UE4Binary('./LinuxNoEditor/RealisticRendering/Binaries/RealisticRendering') # For Linux
    >>> binary = UE4Binary('./MacNoEditor/RealisticRendering.app') # For mac
    >>> with binary: # Automatically launch and close the binary
    >>>     client.request('vget /unrealcv/status')
    >>> # Or alternatively
    >>> binary.start()
    >>> client.request('vget /unrealcv/status')
    >>> binary.close()
    """
    binary_wrapper_selection = {
        'Linux': LinuxBinary,
        'Mac': MacBinary,
        'Win64': WindowsBinary,
    }
    platform_name = get_platform_name()
    binary_wrapper_class = binary_wrapper_selection.get(platform_name)
    if binary_wrapper_class:
        return binary_wrapper_class(binary_path)
    else:
        # Add error handling if needed
        return None


# The environment runner
class UE4BinaryBase(abc.ABC):
    """
    UE4BinaryBase is the base class for all platform-dependent classes, it is
    different from UE4Binary which serves as a factory to create a platform-
    dependent binary wrapper. User should use UE4Binary instead of UE4BinaryBase
    Binary is a python wrapper to control the start and stop of a UE4 binary.
    The wrapper provides simple features to start and stop the binary, mainly
    useful for automate the testing.

    Usage:
        bin = UE4Binary('/tmp/RealisticRendering/RealisticRendering')
        with bin:
            client.request('vget /camera/0/lit test.png')
    """

    def __init__(self, binary_path):
        self.binary_path = binary_path
        self.subproc = None

    def __enter__(self):
        """Start the binary"""
        if os.path.isfile(self.binary_path) or os.path.isdir(self.binary_path):
            self.start()
        else:
            print('Binary %s can not be found' % self.binary_path)

    def __exit__(self, type, value, traceback):
        """Close the binary"""
        self.close()

    @abc.abstractmethod
    def start(self, render_driver, map_name):
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError


class WindowsBinary(UE4BinaryBase):
    def start(self, render_driver=None, map_name='Blank'):
        if self.subproc is not None:
            return

        print('Start windows binary %s' % self.binary_path)
        self.subproc = subprocess.Popen([self.binary_path, map_name])
        print('Waiting for 6 sec.. to make sure env successfully starts')
        time.sleep(6)  # FIXME: How long is needed for the binary to launch?
        # Wait for the process to run. FIXME: Wait for an output line?

        atexit.register(self.__class__.close, self)

    def close(self):
        # Kill windows process
        if self.subproc is None:
            return

        self.subproc.kill()
        self.subproc = None


class LinuxBinary(UE4BinaryBase):
    def start(self, render_driver='vulkan', map_name='Blank'):
        if self.subproc is not None:
            return

        if render_driver == 'opengl4':
            # set render GPU with CUDA_VISIBLE_DEVICES
            print('=>Info: env render with opengl4')
            self.subproc = subprocess.Popen(
                [self.binary_path, map_name, '-opengl4'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif render_driver == 'vulkan':
            # set GPU via CUDA_VISBILE_DEVICES
            print('=>Info: env render with vulkan')
            gpu_ids = os.getenv('CUDA_VISIBLE_DEVICES', '0')
            if gpu_ids == '':
                raise RuntimeError

            gpu_id = list(map(int, map(str.strip, gpu_ids.split(','))))[0]

            self.subproc = subprocess.Popen(
                [
                    self.binary_path,
                    map_name,
                    '-vulkan',
                    '-RenderOffScreen',
                    '-ForceRes',
                    f'-graphicsadapter={gpu_id}',
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            raise NotImplementedError

        binary_sleep_time = int(os.getenv('UE4Binary_SLEEPTIME', '30'))
        print(f'Waiting for {binary_sleep_time} sec.. to make sure env successfully starts')
        time.sleep(binary_sleep_time)

        atexit.register(self.__class__.close, self)

    def close(self):
        # Kill Linux process
        if self.subproc is None:
            return

        self.subproc.kill()
        self.subproc = None


class MacBinary(UE4BinaryBase):
    def start(self, render_driver=None, map_name=None):
        if self.subproc is not None:
            return

        self.subproc = subprocess.Popen(['open', self.binary_path])
        # TODO: Track the stdout to see whether it is started?
        time.sleep(5)

        atexit.register(self.__class__.close, self)

    def close(self):
        # Kill macOS process
        if self.subproc is None:
            return

        self.subproc.kill()
        self.subproc = None


class DockerBinary(UE4BinaryBase):
    def start(self, render_driver=None, map_name=None):
        # nvidia-docker run --rm -p 9000:9000 --env="DISPLAY"
        # --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" qiuwch/rr:${version} >
        # log/docker-rr.log &
        pass

    def close(self):
        pass


if __name__ == '__main__':
    import argparse

    from unrealcv import client

    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', help='Test running the binary', required=True)
    # Example: D:\temp\dev_project_output\WindowsNoEditor\UnrealcvDevProject.exe

    args = parser.parse_args()
    # A hacky way to determine the binary type
    binary_path = args.binary
    if binary_path.lower().endswith('.exe'):
        binary = WindowsBinary(binary_path)
    elif binary_path.lower().endswith('.app'):
        binary = MacBinary(binary_path)
    else:
        binary = LinuxBinary(binary_path)
    with binary:
        client.connect()
        client.request('vget /unrealcv/status')

    pass
    # Try some simple tests in here?

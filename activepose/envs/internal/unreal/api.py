import os
import socket
import sys
from collections import OrderedDict

from .binary import UE4Binary


# api for running unrealenv
class RunUnreal:
    def __init__(self, bin_path):
        self.closed = False
        self.bin_path = bin_path
        self.config_path = os.path.join(os.path.dirname(self.bin_path), 'unrealcv.ini')
        assert os.path.exists(self.bin_path), 'binary file does not exist: %s' % self.bin_path
        assert os.path.exists(self.config_path), 'config file does not exist: %s' % self.config_path
        # default: all parameters are already in this file, some may need modification.
        self.config = self.read_config()

    def start(self, port=9000, resolution=(640, 480), render_driver='vulkan', map_name='Blank'):
        # priority: param > init_file
        env_ip = '127.0.0.1'
        update_config_flag = False
        # write new file if there is no designated port or it is not free.
        if 'port' not in self.config:
            update_config_flag = True

        while not self.isPortFree(env_ip, port):
            port += 1

        if port != int(self.config.get('port', -1)):
            update_config_flag = True
        # always update
        self.config['port'] = str(port)

        # write new file if there is no resolution designated or not equal to new value.
        if 'width' not in self.config or 'height' not in self.config:
            update_config_flag = True
        else:
            width, height = self.config['width'], self.config['height']
            if width != resolution[0] or height != resolution[1]:
                update_config_flag = True
        self.config['width'], self.config['height'] = str(resolution[0]), str(resolution[1])

        if update_config_flag:
            self.write_config_file()

        # self.modify_permission(self.path2env)

        # start env
        self.binary = UE4Binary(self.bin_path)
        self.binary.start(render_driver, map_name)

        if hasattr(self.binary, 'pid'):
            print(f'Env has started, pid:{self.binary.pid}')
        else:
            print('Env has started')

        self.unix_socket_path = os.path.join(os.path.dirname(self.bin_path), f'{port}.socket')
        return env_ip, port, self.unix_socket_path

    def close(self):
        if hasattr(self, 'binary'):
            self.binary.close()
        self.closed = True

    def __del__(self):
        if self.closed:
            return

        self.close()

    def read_config(self):
        config = OrderedDict()
        if os.path.exists(self.config_path):
            with open(self.config_path) as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip('\n').split('=')
                        config[key.lower()] = value
        else:
            print('config file does not exist: %s' % self.config_path)
        return config

    def write_config_file(self):
        line_list = ['[UnrealCV.Core]']
        with open(self.config_path, 'w') as f:
            for k, v in self.config.items():
                line_list.append(k + '=' + v)
            text = '\n'.join(line_list)
            f.write(text)

    def isPortFree(self, ip, port, timeout=1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if 'linux' in sys.platform:
            try:
                sock.bind((ip, port))
                flag = True
            except Exception as e:
                print(e)
                flag = False
            finally:
                sock.close()
                return flag
        elif 'win' in sys.platform:
            sock.settimeout(timeout)
            try:
                sock.connect((ip, port))
                sock.shutdown(socket.SHUT_RDWR)
                flag = False
            except BaseException:
                flag = True
            finally:
                sock.close()
                return flag
        else:
            assert 0, 'not supported system %s' % sys.platform

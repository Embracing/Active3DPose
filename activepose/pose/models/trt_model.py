import numpy as np
import pycuda.driver as cuda

import tensorrt as trt

from . import autoprimaryctx


class ONNXWrapper:
    def __init__(
        self,
        file,
        max_batch_size,
        input_name='input',
        output_name='output',
        target_dtype=np.float32,
    ):
        self.max_batch_size = max_batch_size
        self.input_name = input_name
        self.output_name = output_name
        self.target_dtype = target_dtype

        self.load(file)
        self.allocate_memory()

    def load(self, file):
        f = open(file, 'rb')
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

        engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()

        self.engine_input_idx = engine.get_binding_index(self.input_name)
        self.engine_output_idx = engine.get_binding_index(self.output_name)

        self.fixed_input_shape = tuple(engine.get_binding_shape(self.engine_input_idx))
        self.fixed_input_shape = self.fixed_input_shape[1:]

        self.fixed_output_shape = tuple(engine.get_binding_shape(self.engine_output_idx))
        self.fixed_output_shape = self.fixed_output_shape[1:]

        f.close()

    def allocate_memory(self):
        """
        allocate max size memory
        """
        input_batch = np.empty(
            (self.max_batch_size,) + self.fixed_input_shape, dtype=self.target_dtype
        )
        output_batch = np.empty(
            (self.max_batch_size,) + self.fixed_output_shape, dtype=self.target_dtype
        )
        # Need to set both input and output precisions to FP16 to fully enable FP16

        # Allocate device memory
        self.d_input = cuda.mem_alloc(1 * input_batch.nbytes)
        self.d_output = cuda.mem_alloc(1 * output_batch.nbytes)

        self.bindings = [int(self.d_input), int(self.d_output)]

        self.stream = cuda.Stream()

    def __call__(self, batch):  # result gets copied into output
        """
        Make sure batch size is smaller than max_input_size,
        batch:   numpy ndarray

        Input: np.ndarray  [N, C, H, W]
        Return: np.ndarray  [N, c, h, w]
        """
        input_shape = batch.shape
        batch_szie = input_shape[0]
        output_shape = (batch_szie,) + self.fixed_output_shape

        # print(f'input_shape: {input_shape}')
        # print(f'output_shape: {output_shape}')

        # set dynamic input shape for engine
        self.context.set_binding_shape(self.engine_input_idx, batch.shape)

        # allocate page locked numpy buffer
        input_buffer = cuda.pagelocked_empty(input_shape, dtype=self.target_dtype)
        np.copyto(input_buffer, batch)
        output_buffer = cuda.pagelocked_empty(output_shape, dtype=self.target_dtype)

        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, input_buffer, self.stream)
        # Execute model
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(output_buffer, self.d_output, self.stream)
        # Syncronize threads
        self.stream.synchronize()

        return output_buffer

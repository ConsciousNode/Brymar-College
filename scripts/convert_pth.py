#!/usr/bin/env python3
# scripts/convert_pth.py
# Converts PyTorch .pth / .bin files to .safetensors WITHOUT needing PyTorch installed.
# Uses standard library zipfile and pickle to parse chunked storage.

import sys
import os
import json
import struct
import pickle
import zipfile

def get_dtype(storage_class):
    if storage_class == 'torch.FloatStorage': return 'F32', 4
    if storage_class == 'torch.HalfStorage': return 'F16', 2
    if storage_class == 'torch.BFloat16Storage': return 'BF16', 2
    if storage_class == 'torch.IntStorage': return 'I32', 4
    if storage_class == 'torch.LongStorage': return 'I64', 8
    if storage_class == 'torch.ByteStorage': return 'U8', 1
    if storage_class == 'torch.BoolStorage': return 'BOOL', 1
    raise ValueError(f"Unknown storage class: {storage_class}")

def rebuild_tensor(*args):
    return args

class PthUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'collections' and name == 'OrderedDict':
            from collections import OrderedDict
            return OrderedDict
        if module.startswith('torch'):
            if name == '_rebuild_tensor_v2':
                return rebuild_tensor
            return f"{module}.{name}"
        return super().find_class(module, name)
        
    def persistent_load(self, pid):
        return pid

def convert_pth_to_safetensors(pth_path, out_path):
    print(f"Converting {pth_path} to {out_path}...")
    
    with zipfile.ZipFile(pth_path, 'r') as z:
        # Find data.pkl
        pkl_name = None
        prefix = ""
        for name in z.namelist():
            if name.endswith('data.pkl'):
                pkl_name = name
                prefix = name[:-8] # e.g. 'archive/data.pkl' -> 'archive/'
                break
                
        if not pkl_name:
            raise ValueError("Not a valid PyTorch v1.6+ ZIP archive (no data.pkl found).")
            
        with z.open(pkl_name) as f:
            tensors_dict = PthUnpickler(f).load()
            
        metadata = {"__metadata__": {"format": "pt"}}
        tensor_info = []
        
        current_offset = 0
        for name, args in tensors_dict.items():
            if not isinstance(args, tuple) or len(args) < 4:
                continue
            
            # args is usually (storage_pid, storage_offset, size, stride, ...)
            storage_pid = args[0]
            storage_offset = args[1]
            shape = args[2]
            
            # storage_pid: ('storage', storage_class, key, location, numel)
            if not isinstance(storage_pid, tuple) or storage_pid[0] != 'storage':
                continue
                
            storage_class = storage_pid[1]
            key = storage_pid[2]
            
            dtype, element_size = get_dtype(storage_class)
            
            num_elements = 1
            for dim in shape:
                num_elements *= dim
            
            byte_size = num_elements * element_size
            
            tensor_info.append({
                "name": name,
                "key": key,
                "storage_offset": storage_offset * element_size,
                "dtype": dtype,
                "shape": list(shape),
                "byte_size": byte_size,
                "start": current_offset,
                "end": current_offset + byte_size
            })
            
            metadata[name] = {
                "dtype": dtype,
                "shape": list(shape),
                "data_offsets": [current_offset, current_offset + byte_size]
            }
            
            current_offset += byte_size
            
        # Write safetensors
        with open(out_path, 'wb') as out_f:
            header_json = json.dumps(metadata, separators=(',', ':')).encode('utf-8')
            
            # Pad header to 8 byte alignment
            padding = 8 - (len(header_json) % 8)
            if padding < 8:
                header_json += b' ' * padding
                
            out_f.write(struct.pack('<Q', len(header_json)))
            out_f.write(header_json)
            
            # Write tensor data
            for info in tensor_info:
                data_path = f"{prefix}data/{info['key']}"
                with z.open(data_path) as data_f:
                    # Seek to storage offset
                    data_f.read(info['storage_offset'])
                    
                    # Read tensor bytes
                    bytes_remaining = info['byte_size']
                    while bytes_remaining > 0:
                        chunk = data_f.read(min(bytes_remaining, 1024 * 1024 * 16)) # 16MB chunks
                        if not chunk:
                            raise EOFError(f"Unexpected EOF reading tensor {info['name']}")
                        out_f.write(chunk)
                        bytes_remaining -= len(chunk)
                        
    print(f"Successfully converted to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_pth.py <input.pth> <output.safetensors>")
        sys.exit(1)
    convert_pth_to_safetensors(sys.argv[1], sys.argv[2])

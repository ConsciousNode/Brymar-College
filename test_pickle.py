import pickle
import zipfile

class MockStorage:
    pass

def rebuild_tensor(*args):
    return args

class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'collections' and name == 'OrderedDict':
            from collections import OrderedDict
            return OrderedDict
        if module.startswith('torch'):
            if name == '_rebuild_tensor_v2':
                return rebuild_tensor
            print(f"Loading torch class: {module}.{name}")
            return f"{module}.{name}"
        return super().find_class(module, name)
        
    def persistent_load(self, pid):
        return pid

with zipfile.ZipFile('/home/kimi/Downloads/rwkv7-g1d-0.1b-20260129-ctx8192.pth', 'r') as z:
    pkl_name = [n for n in z.namelist() if n.endswith('data.pkl')][0]
    with z.open(pkl_name) as f:
        data = Unpickler(f).load()
        first_key = list(data.keys())[0]
        print("First item storage class:", data[first_key][0][1])

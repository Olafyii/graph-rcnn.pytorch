import numpy as np
import torch
def inspect(name, x, color=31):
    if type(x) in [list, tuple]:
        print('\033[1;%d;40m%s\033[0m type: %s, len: %s'%(color, name, str(type(x)), str(len(x))))
        inspect('First element of %s'%name, x[0], color)
    elif type(x) == torch.Tensor:
        print('\033[1;%d;40m%s\033[0m type: torch.Tensor, size: %s'%(color, name, str(x.size())))
    elif type(x) == np.ndarray:
        print('\033[1;%d;40m%s\033[0m type: np.ndarray, shape: %s'%(color, name, str(x.shape)))
    elif type(x) in [int, float, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
        print('\033[1;%d;40m%s\033[0m type: %s, value: %s'%(color, name, str(type(x)), x))
    elif type(x) == dict:
        print('\033[1;%d;40m%s\033[0m type: dict, keys: %s'%(color, name, str(x.keys())))
    else:
        print('\033[1;%d;40m%s\033[0m type: %s'%(color, name, str(type(x))))
        try:
            print('\033[1;%d;40mfields of %s: \033[0m'%(color, name), end='')
            print(x.fields())
        except:
            pass
    print()
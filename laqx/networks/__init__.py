from .transformer import make_transformer
from .nnb import make_nnb

from .tensor import make_tensor
from .scale import make_scale
from .ace import make_ace
from .ace_peft import make_ace_peft
from .cnn_mps import make_cnn_mps
from .tensor_nes import make_tensor_nes
from .ace_nes import make_ace_nes



def network_provider(args):
    name = args.network_name
    if name == "transformer":
        return make_transformer(args)
    if name == "nnb":
        return make_nnb(args)
    
    if name == 'tensor':
        return make_tensor(args)
    if name == 'scale':
        return make_scale(args)
    if name == 'ace':
        return make_ace(args)
    
    if name == 'ace_peft':
        return make_ace_peft(args)
    
    if name == 'cnn_mps':
        return make_cnn_mps(args)
    
    if name == 'tensor_nes':
        return make_tensor_nes(args)
    if name == 'ace_nes':
        return make_ace_nes(args)
    
    

    
    raise NotImplementedError

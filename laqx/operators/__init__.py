import jax

from .hamiltonian.hubbard import make_hubbard
from .hamiltonian.hofstadter import make_hofstadter
from .hamiltonian.tri_hofstadter import make_tri_hofstadter
from .hamiltonian.haldane import make_haldane
from .hamiltonian.spin import make_spin
from .hamiltonian.hubbard_alter import make_hubbard_alter

from .hamiltonian.fermi import make_fermi
from .hamiltonian.localo import make_localo
from .hamiltonian.density import make_density
from .hamiltonian.polar import make_polar

from .mcmc import make_mcmc_step
from .apply_operator import make_operator
from .gfmc_utils import get_gfmc_update_fn


def get_operator(args, inference=False):
    if not inference or args.obs == "energy":
        if args.model == "hubbard":
            operator = make_hubbard(args)
        elif args.model == "hofstadter":
            operator = make_hofstadter(args)
        elif args.model == "tri_hofstadter":
            operator = make_tri_hofstadter(args)
        elif args.model == "haldane":
            operator = make_haldane(args)
        elif args.model == "spin":
            operator = make_spin(args)
        elif args.model == "hubbard_alter":
            operator = make_hubbard_alter(args)
        else:
            raise NotImplementedError
    else:
        if args.obs == "fermi":
            operator = make_fermi(args)
        elif args.obs == "localo":
            operator = make_localo(args)
        elif args.obs == "density":
            operator = make_density(args)
        else:
            raise NotImplementedError
    
    return operator


def operator_provider(network, args, inference=False):
    if args.obs == 'polar':
        return make_polar(args)
    operator = get_operator(args, inference)
    local_energy = make_operator(network, operator, args)
    return local_energy

def mcmc_provider(network, args):
    operator = get_operator(args)
    mcmc_step = make_mcmc_step(network, operator, args)
    batch_mcmc_step = jax.vmap(mcmc_step, in_axes=(None, 0, 0))
    return batch_mcmc_step

    
def gfmc_provider(network, args):
    operator = get_operator(args)
    gfmc_update_fn = get_gfmc_update_fn(network, operator, args)
    return gfmc_update_fn
    
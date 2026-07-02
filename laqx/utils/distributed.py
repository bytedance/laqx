import os

import jax


def initialize_distributed_runtime():
    """Initialize JAX distributed runtime using standard environment variables.

    Uses JAX standard env vars: JAX_COORDINATOR_ADDRESS, JAX_NUM_PROCESSES, JAX_PROCESS_ID.
    If not set, returns single-host configuration.
    """
    coordinator_address = os.environ.get('JAX_COORDINATOR_ADDRESS', None)
    num_processes = int(os.environ.get('JAX_NUM_PROCESSES', '1'))
    process_id = int(os.environ.get('JAX_PROCESS_ID', '0'))

    if coordinator_address and num_processes > 1:
        jax.distributed.initialize(coordinator_address, num_processes, process_id)

    return num_processes, process_id

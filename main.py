from fosr.hyperparams import get_args_from_input
from attrdict import AttrDict

default_args = AttrDict({
    "benchmark": "planetoid"
})

args = default_args + get_args_from_input()



if args.benchmark == 'planetoid':
    from examples.planetoid.run_benchmark import run
    run(args)
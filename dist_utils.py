from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import paddle.fluid as fluid


def prepare_for_multi_process(exe, build_strategy, train_prog):
    # prepare for multi-process
    trainer_id = int(os.environ.get('PADDLE_TRAINER_ID', 0))
    num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM'), 1)
    if num_trainers < 2:
        return
    print("PADDLE_TRAINER_ID", trainer_id)
    print("PADDLE_TRAINERS_NUM", num_trainers)

    build_strategy.trainer_id = trainer_id
    build_strategy.num_trainers = num_trainers
    # use multi processes to train the model, and each process use one GPU card
    startup_prog = fluid.Program()
    nccl2_prepare(trainer_id, startup_prog, train_prog)
    # the startup_prog are run two times, but it doesn't matter
    exe.run(startup_prog)


def nccl2_prepare(trainer_id, startup_prog, main_prog):
    config = fluid.DistributeTranspilerConfig()
    config.mode = 'nccl2'
    t = fluid.DistributeTranspiler(config=config)
    t.transpile(trainer_id, trainers=os.environ.get('PADDLE_TRAINER_ENDPOINTS'), current_endpoint=os.environ.get('PADDLE_CURRENT_ENDPOINTS'), startup_program=startup_prog, program=main_prog)
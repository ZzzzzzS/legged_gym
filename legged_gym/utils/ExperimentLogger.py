# record experiment parameters and results
import os
from legged_gym import LEGGED_GYM_ROOT_DIR
from datetime import datetime
from legged_gym.envs.base.base_config import BaseConfig
from legged_gym.utils.helpers import class_to_dict
import yaml


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


def generate_logdir(experiment_name: str):
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", experiment_name)
    log_dir = os.path.join(log_root, datetime.now().strftime("%b%d_%H-%M-%S"))
    print(f"Logging to: {log_dir}")
    return {"log_root": log_root, "log_dir": log_dir}


def commit_experiment(logdir: dict, tsk_arg=None):
    mkdir(logdir["log_dir"])
    filename = os.path.join(logdir["log_dir"], "commit.txt")
    # commit_msg = input("Enter commit message: ")
    commit_msg = "zzs_legged_gym"
    pri_commit_msg = ""
    if tsk_arg is not None:
        pri_commit_msg = f"Task: {tsk_arg.task},  \n"
        if tsk_arg.resume:
            pri_commit_msg += f"Resume from: {tsk_arg.load_run},  \n"

    if commit_msg == "":
        print("No commit message provided! Exiting.")
        os._exit(0)

    out_msg = pri_commit_msg + commit_msg
    with open(filename, "w") as f:
        f.write(out_msg)
    print(f"Commit message saved to {filename}")
    return {"commit message": out_msg}


def save_hyper_params(dir: dict, env_cfg: BaseConfig, train_cfg: BaseConfig):
    if type(env_cfg) is not dict:
        env_cfg = class_to_dict(env_cfg)
    filename = os.path.join(dir["log_dir"], "env_cfg.yaml")
    # save to yaml file
    with open(filename, "w") as f:
        yaml.dump(env_cfg, f)
    print(f"Hyper parameters saved to {filename}")

    if type(train_cfg) is not dict:
        train_cfg = class_to_dict(train_cfg)
    filename = os.path.join(dir["log_dir"], "train_cfg.yaml")
    # save to yaml file
    with open(filename, "w") as f:
        yaml.dump(train_cfg, f)
    print(f"Hyper parameters saved to {filename}")

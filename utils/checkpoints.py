import os
import torch

STATE_DICT_PATH = "state_dicts/"

def save_model_checkpoint(name, global_step,
                          model,
                          optimizer,
                          scheduler,
                          train_loss,
                          train_acc,
                          test_loss=None,
                          test_acc=None,
                          force=False):
    """
    :param param1: this is a first param
    :param force: Overwrite when a checkpoint is already present
    """
    if not os.path.isdir(os.path.join(STATE_DICT_PATH, name)):
        print(
            f"Creating Directory {os.path.isdir(os.path.join(STATE_DICT_PATH, name))}")
        os.mkdir(os.path.join(STATE_DICT_PATH, name))

    save_path = os.path.join(
        STATE_DICT_PATH, f"{name}/eph_{str(global_step).zfill(5)}.pt")
    save_path_model = os.path.join(
        STATE_DICT_PATH, f"{name}/eph_{str(global_step).zfill(5)}.model")

    save_dict = {
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
    }

    if os.path.isfile(save_path) and not force:
        print(f"Checkpoint already exist. Overwriting is disabled. Not saving checkpoint.")
        raise FileExistsError()

    if force:
        print(f"Overwriting checkpoint {save_path}")
    else:
        print(f"Saving checkpoint {save_path}")

    torch.save(save_dict, save_path)
    torch.save(model, save_path_model)


def load_model_checkpoint(name, global_step=None):
    """
    :param global_step: Which global_step to load. If None, it loads the latest global_step
    """
    save_dir = os.path.join(STATE_DICT_PATH, f"{name}")
    if global_step is None:
        checkpoints = os.listdir(save_dir)

        if len(checkpoints) == 0:
            raise FileNotFoundError("No Checkpoint exist.")

        all_global_steps = [int(c.split("_")[-1].split(".")[0]) for c in checkpoints]
        global_step = max(all_global_steps)
        print(f"Loading latest global_step {global_step}")

    save_path = os.path.join(
        STATE_DICT_PATH, f"{name}/eph_{str(global_step).zfill(5)}.pt")
    save_path_model = os.path.join(
        STATE_DICT_PATH, f"{name}/eph_{str(global_step).zfill(5)}.model")

    checkpoint = torch.load(save_path)
    model = torch.load(save_path_model)
    print(f"Loaded Checkpoint \n \
        | global_step {checkpoint['global_step']} \
        | train_loss {checkpoint['train_loss']} | train_acc {checkpoint['train_acc']} \
        | test_acc {checkpoint['test_acc']} | test_acc {checkpoint['test_acc']}")
    return checkpoint, model

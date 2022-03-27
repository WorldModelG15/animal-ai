import sys, os
from animalai.envs.environment import AnimalAIEnvironment

import torch
from main import Trainer

from wrapper import WrapPyTorch, OneHotAction, MaxAndSkipEnv, DummyWrapper
from gym_unity.envs import UnityToGymWrapper


if __name__ == "__main__":
    if len(sys.argv) > 1:
        configuration_file = sys.argv[1]
    else:
        tasks_folder = "/root/mnt/animalai/animal-ai/examples/tasks/"
        configuration_files = os.listdir(tasks_folder)
        configuration_file_paths = []
        for file_name in configuration_files:
            configuration_file_paths.append(tasks_folder + file_name)
    aai_env = AnimalAIEnvironment(
        seed=123,
        file_name="/root/mnt/animalai/animal-ai/env/AnimalAI",
        arenas_configurations_paths=configuration_file_paths,
        play=False,
        base_port=5020,
        inference=False,
        useCamera=True,
        resolution=64,
        useRayCasts=False,
        no_graphics=True,
        # raysPerSide=1,
        # rayMaxDegrees = 30,
    )
    env = UnityToGymWrapper(
        aai_env, uint8_visual=True, allow_multiple_obs=False, flatten_branched=True
    )
    # env = OneHotAction(MaxAndSkipEnv(env))
    env = OneHotAction(MaxAndSkipEnv(env))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    trainer = Trainer(env, device, train=False)
    trainer.load_models(
        "/root/mnt/animalai/animal-ai/examples/models/20220325112323/episode_1000"
    )
    trainer.view(10)

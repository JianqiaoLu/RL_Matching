from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Any, Optional
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


@dataclass
class TrainConfig:
    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the AdamW optimizer
    BATCH_SIZE : int
    GAMMA : float
    EPS_START : float
    EPS_END : float
    EPS_DECAY : int
    TAU : float
    LR : float
    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)



@dataclass
class RL_TrainConfig:
    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the AdamW optimizer
    BATCH_SIZE : int
    SAVE_PATH: str
    SAVE_INTERVAL: int
    LR : float
    NUM_EPS : int
    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)

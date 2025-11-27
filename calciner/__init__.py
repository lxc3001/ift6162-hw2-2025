"""
Flash Calciner Control Package

This package provides:
- Physics-based simulation of a flash clay calciner
- Neural surrogate model for fast dynamics evaluation
- RL environment for the simplified 1D control problem
"""

from .physics import SimplifiedFlashCalciner, N_SPECIES, L
from .surrogate import (
    CalcinerSimulator,
    SpatiallyAwareDynamics,
    SurrogateModel,
    TransitionDataset,
    generate_training_data,
)
from .mpc import CalcinerDynamics
from .baselines import CalcinerEnv, ConstantTemperatureController, evaluate_baseline

__all__ = [
    # Physics
    'SimplifiedFlashCalciner',
    'N_SPECIES',
    'L',
    # Surrogate
    'CalcinerSimulator',
    'SpatiallyAwareDynamics', 
    'SurrogateModel',
    'TransitionDataset',
    'generate_training_data',
    # Simplified dynamics
    'CalcinerDynamics',
    # RL Environment
    'CalcinerEnv',
    'ConstantTemperatureController',
    'evaluate_baseline',
]

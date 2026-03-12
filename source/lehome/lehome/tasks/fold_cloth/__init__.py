"""Garment folding task — Manager-based environment.

Registers:
  - LeHome-BiSO101-ManagerBased-Garment-v0: Hybrid manager-based env with
    manual GarmentObject particle cloth management.
  - LeHome-BiSO101-ManagerBased-Garment-Mimic-v0: Same env with MimicGen
    subtask definitions for data augmentation.
"""
import gymnasium as gym

gym.register(
    id="LeHome-BiSO101-ManagerBased-Garment-v0",
    entry_point=f"{__name__}.garment_fold_env:GarmentFoldEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fold_cloth_bi_arm_env_cfg:GarmentFoldEnvCfg",
    },
)

gym.register(
    id="LeHome-BiSO101-ManagerBased-Garment-Mimic-v0",
    entry_point=f"{__name__}.garment_fold_env:GarmentFoldEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fold_cloth_bi_arm_mimic_env_cfg:GarmentFoldMimicEnvCfg",
    },
)

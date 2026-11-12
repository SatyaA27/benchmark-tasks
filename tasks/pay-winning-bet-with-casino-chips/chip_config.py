"""Lazy Isaac config factory in a normally imported, registered module."""


def chip_config():
    import isaaclab.sim as sim
    from isaaclab.utils import configclass

    # task_file.load() does not register its task module in sys.modules.
    # configclass merges SpawnerCfg's string annotations before dataclass looks
    # up the subclass module, so define the subclasses in this imported helper.
    @configclass
    class ChipCfg(sim.SpawnerCfg):
        chip_name: str = ""

    @configclass
    class FeltCfg(sim.SpawnerCfg):
        felt_name: str = ""

    return ChipCfg, FeltCfg

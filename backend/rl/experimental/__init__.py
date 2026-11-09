"""Experimental DQN variants. Each lives in its own module and changes exactly one mechanism of
backend.rl.agents.dqn.DQNAgent. Register new variants in VARIANTS (name -> "module:Class").
"a+b" composes variants through cooperative multiple inheritance (used for hybrids and ablations)."""
from __future__ import annotations

import hashlib
import importlib
import inspect
from pathlib import Path

_P = "backend.rl.experimental."
VARIANTS: dict[str, str] = {
    "control": "backend.rl.agents.dqn:DQNAgent",
    "double": _P + "double:DoubleDQN",
    "nstep": _P + "nstep:NStep",
    "per": _P + "per:PrioritizedReplay",
    "softtarget": _P + "softtarget:SoftTarget",
    "dueling": _P + "dueling:Dueling",
    "aux_dynamics": _P + "aux_dynamics:AuxDynamics",
    "prior_explore": _P + "prior_explore:PriorExplore",
    "collapse_guard": _P + "collapse_guard:CollapseGuard",
    "crash_focus": _P + "crash_focus:CrashFocus",
}


def _one(name: str):
    module, cls = VARIANTS[name].split(":")
    return getattr(importlib.import_module(module), cls)


def parts(name: str) -> list[str]:
    names = name.split("+")
    unknown = [n for n in names if n not in VARIANTS]
    if unknown or (len(names) > 1 and "control" in names) or len(set(names)) != len(names):
        raise ValueError(f"unknown or invalid variant {name!r}; known: {sorted(VARIANTS)}")
    return names


def variant_class(name: str):
    classes = [_one(n) for n in parts(name)]
    return classes[0] if len(classes) == 1 else type("Combo_" + "_".join(parts(name)), tuple(classes), {"name": name})


def source_sha(name: str) -> str:
    """Hash of the variant's source file(s) + the base agent: identifies the exact algorithm code of a run."""
    files = sorted({inspect.getsourcefile(_one(n)) for n in [*parts(name), "control"]})
    return hashlib.sha256(b"".join(Path(f).read_bytes() for f in files)).hexdigest()[:12]


def make_agent(name: str, cfg, total_steps: int, device: str = "cpu", seed: int = 0):
    return variant_class(name)(cfg, total_steps, device, seed)

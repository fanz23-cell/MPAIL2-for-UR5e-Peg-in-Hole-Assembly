"""Clean evaluation overrides for OmniReset Isaac Lab configs."""

from __future__ import annotations

from typing import Any


def _set_param_if_present(term: Any, key: str, value: Any) -> bool:
    params = getattr(term, "params", None)
    if not isinstance(params, dict) or key not in params:
        return False
    params[key] = value
    return True


def apply_omnireset_clean_eval_overrides(env_cfg: Any) -> dict[str, list[str]]:
    """Disable sim2real randomization while preserving fixed controller setup.

    OmniReset names several eval controller events like randomization terms
    (notably ``randomize_osc_gains``), but these events also install the fixed
    controller/sysid parameters used by the official finetune/eval configs.
    Removing them changes the control behavior.  For clean evaluation we only
    remove physical domain randomization and make controller events deterministic.
    """

    events = getattr(env_cfg, "events", None)
    if events is None:
        raise ValueError("Requested OmniReset clean eval, but env_cfg has no events config.")

    disable_terms = (
        "robot_material",
        "insertive_object_material",
        "receptive_object_material",
        "table_material",
        "randomize_robot_mass",
        "randomize_insertive_object_mass",
        "randomize_receptive_object_mass",
        "randomize_table_mass",
    )
    fixed_terms: list[str] = []
    disabled_terms: list[str] = []

    for term_name in disable_terms:
        if hasattr(events, term_name) and getattr(events, term_name) is not None:
            setattr(events, term_name, None)
            disabled_terms.append(term_name)

    gripper_term = getattr(events, "randomize_gripper_actuator_parameters", None)
    if gripper_term is not None:
        changed = False
        changed |= _set_param_if_present(gripper_term, "stiffness_distribution_params", (1.0, 1.0))
        changed |= _set_param_if_present(gripper_term, "damping_distribution_params", (1.0, 1.0))
        changed |= _set_param_if_present(gripper_term, "distribution", "uniform")
        if changed:
            fixed_terms.append("randomize_gripper_actuator_parameters")

    arm_term = getattr(events, "randomize_arm_sysid", None)
    if arm_term is not None:
        changed = False
        changed |= _set_param_if_present(arm_term, "scale_range", (1.0, 1.0))
        changed |= _set_param_if_present(arm_term, "delay_range", (0, 0))
        if changed:
            fixed_terms.append("randomize_arm_sysid")

    osc_term = getattr(events, "randomize_osc_gains", None)
    if osc_term is not None:
        changed = _set_param_if_present(osc_term, "scale_range", (1.0, 1.0))
        if changed:
            fixed_terms.append("randomize_osc_gains")

    return {"disabled": disabled_terms, "fixed": fixed_terms}

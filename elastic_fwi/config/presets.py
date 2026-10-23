from __future__ import annotations

from elastic_fwi.config.base import CampaignPreset, RunPreset


RUN_PRESETS: dict[str, RunPreset] = {
    "quick": RunPreset(
        name="quick",
        description="Small implementation check; not scientifically conclusive.",
        quick=True,
        allow_heavy_required=False,
    ),
    "medium": RunPreset(
        name="medium",
        description="Intermediate validation scale.",
        quick=False,
        allow_heavy_required=True,
    ),
    "article": RunPreset(
        name="article",
        description="Article-scale production configuration.",
        quick=False,
        allow_heavy_required=True,
    ),
}


CAMPAIGN_PRESETS: dict[str, CampaignPreset] = {
    "quick": CampaignPreset(
        name="quick",
        domain_length=10.0,
        nx=8,
        ny=8,
        num_steps=20,
        every_k=2,
        dt=0.05,
        source_frequency=1.0,
        defect_radius=0.5,
    ),
    "medium": CampaignPreset(
        name="medium",
        domain_length=20.0,
        nx=48,
        ny=48,
        num_steps=80,
        every_k=4,
        dt=0.05,
        source_frequency=1.0,
        defect_radius=0.5,
    ),
    "article": CampaignPreset(
        name="article",
        domain_length=20.0,
        nx=200,
        ny=200,
        num_steps=400,
        every_k=5,
        dt=0.02,
        source_frequency=1.0,
        defect_radius=0.5,
    ),
}


def get_run_preset(name: str) -> RunPreset:
    try:
        return RUN_PRESETS[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown run preset {name!r}. Expected one of {sorted(RUN_PRESETS)}."
        ) from exc


def get_campaign_preset(name: str) -> CampaignPreset:
    try:
        return CAMPAIGN_PRESETS[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown campaign preset {name!r}. Expected one of {sorted(CAMPAIGN_PRESETS)}."
        ) from exc

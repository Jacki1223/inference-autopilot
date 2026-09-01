from pathlib import Path

from setuptools import setup


VERSION = (Path(__file__).resolve().parent / "VERSION").read_text(
    encoding="utf-8"
).strip()


setup(
    name="inference-autopilot",
    version=VERSION,
    description="Private, evidence-driven SGLang inference optimization",
    python_requires=">=3.9",
    packages=["inference_autopilot_data"],
    package_dir={"": "scripts", "inference_autopilot_data": "references"},
    package_data={"inference_autopilot_data": ["*.json"]},
    py_modules=[
        "inferopt",
        "inferopt_cli",
        "autopilot",
        "autotune",
        "profile_sglang",
        "sglang_catalog",
        "sglang_runtime",
        "generate_moe_config",
        "trial_store",
        "bayesian",
        "optimization_rules",
        "parameter_evolution",
        "check_sglang_compat",
        "candidate_registry",
        "mechanism_search",
    ],
    entry_points={"console_scripts": ["inferopt=inferopt_cli:main"]},
)

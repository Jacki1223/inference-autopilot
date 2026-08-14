from setuptools import setup


setup(
    name="inference-autopilot",
    version="0.1.3",
    description="Private, evidence-driven SGLang inference optimization",
    python_requires=">=3.9",
    packages=["inference_autopilot_data"],
    package_dir={"": "scripts", "inference_autopilot_data": "references"},
    package_data={"inference_autopilot_data": ["hardware-profiles.json"]},
    py_modules=[
        "inferopt",
        "inferopt_cli",
        "autopilot",
        "autotune",
        "profile_sglang",
        "sglang_catalog",
        "sglang_runtime",
    ],
    entry_points={"console_scripts": ["inferopt=inferopt_cli:main"]},
)

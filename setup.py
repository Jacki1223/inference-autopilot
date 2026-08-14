from setuptools import setup


setup(
    name="inference-autopilot",
    version="0.1.0",
    description="Private, evidence-driven SGLang inference optimization",
    python_requires=">=3.9",
    package_dir={"": "scripts"},
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

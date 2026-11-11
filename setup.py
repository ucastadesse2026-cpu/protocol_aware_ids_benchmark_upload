"""Minimal packaging. Scripts are meant to be run from the repository root:
    python experiments/run_baseline.py --config configs/default.yaml
"""
from setuptools import find_packages, setup

setup(
    name="protocol_aware_ids_benchmark",
    version="0.1.0",
    description="Protocol-aware adversarial robustness benchmark for satellite IDS (CCSDS-conformant attacks)",
    packages=find_packages(exclude=("tests",)),
    py_modules=["config", "make_tables"],
    python_requires=">=3.10",
    install_requires=["numpy", "pandas", "scipy", "scikit-learn", "pyyaml", "matplotlib"],
    extras_require={"full": ["torch", "adversarial-robustness-toolbox"]},
)

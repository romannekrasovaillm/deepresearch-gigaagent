from setuptools import setup, find_packages

setup(
    name="deepresearch-agent",
    version="0.1.0",
    description="Deep Research Agent for Scientific Synthesis with Code Reasoning",
    author="DeepResearch Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "python-docx>=1.1.0",
        "pandas>=2.2.0",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": ["pytest", "pytest-asyncio", "black", "ruff"],
        "training": ["accelerate", "peft", "flash-attn"],
    },
    entry_points={
        "console_scripts": [
            "dr-convert=scripts.convert_library:main",
            "dr-generate=scripts.generate_dataset:main",
            "dr-train-sft=scripts.train_sft:main",
            "dr-train-rl=scripts.train_rl:main",
        ],
    },
)

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="askme-voice-assistant",
    version="1.0.0",
    author="AskMe Development Team",
    author_email="dev@askme-assistant.com",
    description="Privacy-focused offline voice assistant with custom LLM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/askme/askme-voice-assistant",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.7.0",
        ],
        "training": [
            "accelerate>=0.24.0",
            "bitsandbytes>=0.41.0",
            "peft>=0.7.0",
            "trl>=0.7.0",
            "datasets>=2.14.0",
            "wandb>=0.16.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "askme=src.main:main",
            "askme-setup=scripts.setup_models:main",
            "askme-verify=scripts.verify_installation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "askme": ["configs/*.yaml", "docs/*.md"],
    },
)

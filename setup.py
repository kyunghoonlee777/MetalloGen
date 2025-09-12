import setuptools
from pathlib import Path

long_description = Path("README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="MetalloGen",
    version="0.0.1",
    description="A package for generating metal complexes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD-3-Clause",
    license_files=["LICENSE"],
    author="Kyunghoon Lee et al.",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "cclib",
        "rdkit",
    ],
    entry_points={
        "console_scripts": [
            # metallogen command line tool
            "metallogen=MetalloGen.run:main",
        ],
    },
)
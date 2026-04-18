from setuptools import setup, find_packages

setup(
    name="plato-torch",
    version="0.5.0a1",
    description="PLATO self-training rooms — 21 AI training methods",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    license="MIT",
)

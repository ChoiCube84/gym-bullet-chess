from setuptools import setup, find_packages

setup(
    name="gym_bullet_chess",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["gymnasium>=1.0.0", "chess>=1.10.0", "numpy>=1.24.0"],
    extras_require={"gui": ["pygame"]},
)

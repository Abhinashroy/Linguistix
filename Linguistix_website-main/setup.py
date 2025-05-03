from setuptools import setup, find_packages

setup(
    name="ml_website",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "Flask==2.2.3",
        "Werkzeug==2.2.3",
        "Jinja2==3.1.2",
        "itsdangerous==2.1.2",
        "click==8.1.3",
        "torch==2.2.1",
        "numpy==1.24.2",
        "librosa==0.10.0",
        "SoundFile==0.12.1",
        "matplotlib==3.7.1",
        "scipy==1.10.1",
        "scikit-learn==1.2.2",
    ],
    python_requires=">=3.8",
)
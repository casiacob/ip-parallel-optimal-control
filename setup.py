from setuptools import setup

setup(
    name="noc",
    version="0.0.1",
    author="",
    author_email="",
    description="",
    install_requires=["jax", "matplotlib", "pandas", "jaxlib==0.4.20+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"],
    zip_safe=False,
)

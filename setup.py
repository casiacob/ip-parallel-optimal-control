from setuptools import setup

setup(
    name="noc",
    version="0.0.1",
    author="Casian Iacob",
    author_email="casian.iacob@aalto.fi",
    description="",
    install_requires=["jax[cuda12]", "matplotlib", "pandas"],
    zip_safe=False,
)

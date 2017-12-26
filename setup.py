from setuptools import setup

setup(name="experiments",
      version="0.0.1",
      description="Small wrapper framework for running experiments using Tensorflow's tf.estimator API.",
      long_description=open("README.md").readlines()[1].strip(),
      license="MIT",
      packages=["experiments"],
      install_requires=[r.strip() for r in open("requirements.txt").readlines()])

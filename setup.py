from setuptools import find_packages,setup


setup(name="e-commerce-bot",
      version="0.1",
      author="nikhilsai",
      author_email="nikhilsai064@gmail.com",
      packages=find_packages(),
      install_requires=['langchain-astradb','langchain'])
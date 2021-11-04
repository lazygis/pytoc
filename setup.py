import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytoc",
    version="0.0.2",
    author="Zhen Liu",
    author_email="zhenliu26@outlook.com",
    description="A Python library for generating Total Operating Characteristic (TOC) Curves.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lazygis/pytoc",
    packages=setuptools.find_packages(),
    install_requires=['matplotlib', 'numpy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="Auriga",
    version="1.0",
    author="Marina Kounkel",
    author_email="marina.kounkel@unf.edu",
    description="A neural network for structure parameter determination",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mkounkel/Auriga",
    packages=['auriga', 'auriga.test'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={
        'auriga': ['auriga.pt'],'auriga.test':['test.fits','test.csv']
    },
    install_requires=['torch','torchvision','astropy','astroquery','pandas'],
    entry_points={'console_scripts': ['auriga=auriga.auriga:main']},
    python_requires='>=3.6',
)

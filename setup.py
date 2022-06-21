import setuptools
from tal import __version__ as tal_version

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="y-tal",
    version=tal_version,
    author="Diego Royo & Pablo Luesia",
    author_email="droyo@unizar.es",
    description="(Your) Transient Auxiliary Library - Analysis and processing of time-resolved light transport",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/diegoroyo/tal",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    package_data={
        'tal': [
            '.tal.conf.example',
            'render/scene_defaults.yaml',
        ],
    },
    entry_points={
        'console_scripts': [
            'tal=tal.__main__:main'
        ]
    },
    python_requires=">=3.6",
)

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GridMapDecompose",
    version="0.1.4",
    author="Tomasz Kucner",
    author_email="tomasz.kucner@oru.se",
    description="Simple pacakge for grid map decomposition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={'Source': 'https://github.com/tkucner/GridMapDecompose',
                  'Author': 'https://mro.oru.se/people/tomasz-kucner/',
                  'Tracker': 'https://github.com/tkucner/GridMapDecompose/issues',
                  },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires = ['numpy>=1.16.4',
                        'scipy>=1.2.1',
                        'scikit-image>=0.15.0',
                        'networkx>=2.3',
                        ],
)

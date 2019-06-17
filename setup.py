import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GridMapDecompose",
    version="0.0.1",
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
    install_requires=['numpy',
                      'scipy',
                      'scikit-image',
                      ],
)

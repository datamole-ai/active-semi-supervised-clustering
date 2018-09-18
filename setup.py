import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="active-semi-supervised-clustering",
    version="0.0.1",
    author="Jakub Svehla",
    author_email="jakub.svehla@datamole.cz",
    description="Active semi-supervised clustering algorithms for scikit-learn",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/datamole-ai/active-semi-supervised-clustering",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'metric-learn>=0.4',
    ]
)

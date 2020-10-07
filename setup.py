import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="effmap_demo",
    version="0.0.1",
    author="Ivan Okhotnikov",
    author_email="ivan.okhotnikov@outlook.com",
    description="Streamlit demonstration of usage of regressor and HST object introduced in effmap",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ivanokhotnikov/effmap_demo/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True
)

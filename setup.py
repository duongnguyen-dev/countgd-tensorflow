from setuptools import setup, find_packages

setup(
    name="countgd-tensorflow",
    version="0.1.0",
    author="Nguyen Hai Duong",
    author_email="duongng2911@gmail.com",
    description="Implementation of CountGD using TensorFlow",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/duongnguyen-dev/countgd-tensorflow",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>2.17.0,<=2.18.0",
        "tf-keras>2.17.0,<=2.18.0",
        "tf-models-official==2.18.0",
        "tensorflow-metal==1.2.0",
        "tensorflow-hub==0.16.1",
        "tensorflow-text>2.17.0,<=2.18.0",
        "opencv-python==4.11.0.86",
        "matplotlib==3.10.1",
        "pytest==8.3.5"
    ],
    license="Apache-2.0",
    license_files=["LICENSE"],
    project_urls={
        "Repository": "https://github.com/duongnguyen-dev/countgd-tensorflow",
        "Bug Tracker": "https://github.com/duongnguyen-dev/countgd-tensorflow/issues",
    },
)
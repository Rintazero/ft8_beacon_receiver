from setuptools import setup, find_packages

setup(
    name="FT8_BEACON_RECEIVER",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # 你的项目依赖
    ],
    author="Rintazero",
    author_email="CIRCODE@126.com",
    description="description",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Rintazero/ft8_beacon_receiver",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
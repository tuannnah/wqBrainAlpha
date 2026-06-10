from setuptools import setup, find_packages


setup(
    name="Alpha_Tool",
    version="1.0.0",
    author="YourName",
    description="WorldQuant Brain Alpha Generator",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.31.0",
        "pandas>=2.0.0",
        "lark>=1.1.9,<2",
    ],
    entry_points={
        'console_scripts': [
            'alpha_tool=main:main',
        ],
    },
    # Bao gồm các file không phải Python.
    package_data={
        '': ['*.txt', '*.json', '*.ico', '*.png'],
    },
    # Thông tin bổ sung cho package.
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

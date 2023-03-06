import setuptools

setuptools.setup(
    name='plinio',
    version='0.0.1',
    packages=setuptools.find_packages(),
    install_requires=['setuptools'],
    maintainer='Daniele Jahier Pagliari',
    maintainer_email='daniele.jahier@polito.it',
    description='A simple library for lightweight NAS based on PyTorch',
    license='Apache License 2.0',
    tests_require=['unittest'],
    python_requires=">=3.7",
)

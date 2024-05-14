from setuptools import setup

setup(
    name='tritionfm',
    version='0.1',
    author='hanhui',
    author_email='clearhanhui@gmail.com',
    description='An OpenAI Triton Factorization-Machine implementation.',
    packages=['tritonfm'],
    install_requires=[
        'torch',
    ],
)
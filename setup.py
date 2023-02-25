from setuptools import setup


setup(
    name='hrl_nav',
    description='',
    version='1.0',
    # license='MIT',
    author='Kyowoon Lee, Seongun Kim',
    author_email='leekwoon@unist.ac.kr, seongun@kaist.ac.kr',
    packages=['rlkit', 'hrl_nav'],
    package_dir={'':'src'},
    install_requires=[
        'torch==1.4.0',
        'torchvision==0.5.0',
        # rlkit related
        'gtimer'
    ]
)
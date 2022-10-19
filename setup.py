from setuptools import setup


try:
    # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:
    # for pip <= 9.0.3
    from pip.req import parse_requirements


def load_requirements(fname):
    reqs = parse_requirements(fname, session="test")
    return [str(ir.req) for ir in reqs]


setup(
    name='stochastic_offline_envs',
    version='1.0',
    description='',
    author='Keiran Paster',
    author_email='keirp@cs.toronto.edu',
    packages=['stochastic_offline_envs'],
    install_requires=load_requirements('requirements.txt'),
)

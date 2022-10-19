from setuptools import setup


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

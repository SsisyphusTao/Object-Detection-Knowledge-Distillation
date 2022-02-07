from setuptools import setup, find_packages
from setuptools.command.install import install
from typing import List


def parse_requirements(file_name: str) -> List[str]:
    with open(file_name, encoding='utf-8') as f:
        return [require.strip() for require in f if require.strip() and not require.startswith("#")]


setup(
    name='odkd',
    version='0.0.1',
    description='Object Detection Knowledge Distillation',
    author='Chandler',
    install_requires=parse_requirements('requirements.txt'),
    packages=find_packages(),
    package_data={'odkd.utils': ['template.yml']},
    entry_points={'console_scripts': [
        'odkd-train = odkd:run_train', 'odkd-eval = odkd:run_eval']},
)

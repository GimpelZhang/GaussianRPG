from setuptools import find_packages, setup

package_name = 'simulator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='junchuan',
    maintainer_email='zjunchuan@gmail.com',
    description='The simulator main frame',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simulator = simulator.simulator:main',
            'ground_truth = simulator.groundtruth:main',
            'evaluation = simulator.evaluation:main'
        ],
    },
)

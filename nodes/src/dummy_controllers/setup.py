from setuptools import find_packages, setup

package_name = 'dummy_controllers'

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
    description='A dummy AEB controller with simple logic',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aeb_controller = dummy_controllers.AEB_controller:main'
        ],
    },
)

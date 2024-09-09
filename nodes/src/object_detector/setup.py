from setuptools import find_packages, setup

package_name = 'object_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=['object_detector', 'models', 'networks', 'utils', 'weights'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='junchuan',
    maintainer_email='zjunchuan@gmail.com',
    description='Render images and publish the object distance using yolov5',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detector = object_detector.object_detector:main'
        ],
    },
)

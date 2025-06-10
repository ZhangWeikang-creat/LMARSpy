from setuptools import setup, find_packages

setup(
    name='lmarspy',
    version='1.0',
    packages=find_packages(), 
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'lmarspy=lmarspy.model.main:main',
        ],
    },
    tests_require=['pytest'],
    test_suite='tests',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown', 
    author='Chen Research Group',
    author_email='xichen.me@outlook.com',
    developer = "Zhang Weikang",
    maintainer = "Zhang Weikang",
    developer_email = "weikang-zhang@foxmail.com",
    url='https://github.com/ZhangWeikang-creat/LMARSpy.git',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)

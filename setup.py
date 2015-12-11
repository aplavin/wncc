from distutils.core import setup

setup(
    name='wncc',
    version='1.0',
    packages=['wncc'],
    requires=['numpy', 'planfftw'],
    extras_require={'pyfftw': ['pyfftw']},
    tests_require=['pytest', 'hypothesis', 'hypothesis-numpy'],
    url='https://github.com/aplavin/wncc',
    license='MIT',
    author='Alexander Plavin',
    author_email='alexander@plav.in',
    description='Weighted normalized cross-correlation.',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Software Development :: Libraries'
    ]
)

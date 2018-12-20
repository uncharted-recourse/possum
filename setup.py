from distutils.core import setup

setup(name='Possum',
    version='1.0.0',
    description='Possum - Post Summarization',
    packages=['Possum'],
    install_requires=['sumy >= 0.7.0',
        'scipy >= 0.19.0',
        'numpy>=1.14.2',
        'scikit-learn>=0.18.1',
        'pandas>=0.19.2'
        'nltk>=3.3'],
    include_package_data=True,
)
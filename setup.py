from distutils.core import setup

setup(name='Possum',
    version='1.1.3',
    description='Possum - Post Summarization',
    packages=['Possum'],
    install_requires=['sumy',
        'scipy==1.1.0',
        'numpy>=1.14.2',
        'scikit-learn>=0.18.1',
        'pandas>=0.19.2'
        'nltk>=3.3',
        'regex==2017.4.5',
        'pycountry']
    include_package_data=True,
)
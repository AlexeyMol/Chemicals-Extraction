from setuptools import setup, find_packages
root_package = "mol_analyzer"
print(find_packages())
setup(
    name='chemical_extraction',
    version='1.0.5',
    license='MIT',
    packages=[root_package] + [f"{root_package}.{item}" for item in find_packages(root_package)],
    package_data= {
        # all .dat files at any package depth
        '': ['./GNNT/ckp/*.json', './GNNT/ckp/*.pt'],
    },
    include_package_data=True,
    install_requires=[
    'aiohttp==3.9.5',
    'beautifulsoup4',
    'aiosignal==1.3.1',
    'attrs==23.2.0',
    'certifi==2024.6.2',
    'charset-normalizer==3.3.2',
    'click==8.1.7',
    'filelock==3.15.4',
    'frozenlist==1.4.1',
    'fsspec==2024.6.1',
    'huggingface-hub==0.23.4',
    'idna==3.7',
    'MarkupSafe==2.1.5',
    'mpmath==1.3.0',
    'multidict==6.0.5',
    'networkx==3.3',
    'nltk==3.8.1',
    'numpy==1.26.4',
    'pandas==2.2.2',
    'pillow==10.4.0',
    'psycopg2',
    'psutil==6.0.0',
    'pyparsing==3.1.2',
    'python-dateutil==2.9.0.post0',
    'pytz==2024.1',
    'PyYAML==6.0.1',
    'rdkit==2024.3.1',
    'regex==2024.5.15',
    'requests==2.32.3',
    'safetensors==0.4.3',
    'scikit-learn==1.5.1',
    'scipy==1.14.0',
    'six==1.16.0',
    'soupsieve==2.5',
    'sympy==1.12.1',
    'threadpoolctl==3.5.0',
    'tokenizers==0.19.1',
    'torch==2.3.1',
    'torch_geometric==2.5.3',
    'tqdm==4.66.4',
    'transformers==4.42.3',
    'triton==2.3.1',
    'typing_extensions==4.12.2',
    'tzdata==2024.1',
    'urllib3==2.2.2',
    'yarl==1.9.4'
    ],
)

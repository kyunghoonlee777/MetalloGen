import setuptools

with open('README.md', 'r', encoding = 'utf-8') as fh:
    long_description = fh.read()
    
setuptools.setup(
    name = 'ACE_OM',
    version = '0.0.1',
    description = 'A package for generating metal complexes',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    license = 'BSD 3-Clause License',
    author = 'Minseong Park et al.',
    packages = setuptools.find_packages(),
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    install_requires = [
        'numpy',
        'rdkit',
        'scipy',
        'cclib'
    ],
    python_requires = '>=3.9'
)

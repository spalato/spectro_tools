from setuptools import setup, find_packages

setup(name='spectro',
      version='0.1',
      description='Spectroscopy data analysis tools',
      url='',
      author='S. Palato',
      author_email='',
      license='',
      packages=find_packages("src"),
      package_dir={"": "src"},
      zip_safe=False)
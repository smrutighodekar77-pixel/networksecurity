from setuptools import find_packages,setup
from typing import List


    
def get_requirements() -> List[str]:
    """This function will return List of requirements."""
    
    requirement_list = []
    try:
        with open('requirements.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                # Ignore empty lines and '-e .'
                if requirement and requirement != '-e .':
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print('requirements.txt file not found.')
    return requirement_list

setup(
    name="NetworkSecurity",
    version="0.0.1",
    author="Smruti",
    author_email="smrutighodekar77@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)
   
    
  
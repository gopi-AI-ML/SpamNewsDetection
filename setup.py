from setuptools import setup,find_packages

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str):
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name          =   "SpamNewsLib",
    version       =   "0.0.1",
    author        =   "gopi",
    author_email  =   "gopiaiml@gmail.com",
    packages= find_packages(),
    install_requries = get_requirements("requirements.txt")
)
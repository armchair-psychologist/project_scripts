# All commands needs to be ran from root directory of repo
## To install
chmod +x install.bash && ./install.bash
## To activate venv
source bin/activate
## To deactivate venv
deactivate
## To start(venv needs to be activated)
jupyter notebook
## When adding new dependency(venv needs to be activated)
pip freeze > requirements.txt
## When installing new dependency(need to done after pulls that modify requirement.txt and venv needs to be activated for this)
pip install -r requirements.txt
## Relevant directories
All project related scripts are in project_scripts

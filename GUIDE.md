# Technical Project Template

Continuous documentation and reporting for technical projects.

## Setting up your project

Start by installing Cookiecutter by running `pip install cookiecutter`. Then:

1. Set up a new project. You will be asked for its name, and that of its Python source directory.
   
   `cookiecutter https://github.com/alopezrivera/template_technical`
2. Edit `docs/source/project.py` and `docs/source/project.sty` with the details of your project.
3. Turn your project into a git repository.
   
   `git init`
4. Push your project's initial commit to GitHub.
5. Set up the host for your documentation site. Your options:
   - Create a branch in your repository to host it, and introduce its name in `config.env`
   - Create an external repository to host it, and introduce its details in `config.env`. In this case, you will need to have write permission to the repo, and create a **secret** in your repository with a personal API token (check your profile -> developer settings -> keys).
6. Push again. All is set.

If you go to the Actions tab of your repo, you will see
- An action which just failed (as your site host was not set up). Don't mind it. 
- An action runningm, with two jobs: one for the documentation site and another generating a technical report.

## Important files

### `config.env`

Sets terminal environment variables for the 
site and report generation and upload process.

### `requirements.txt`

Requirements of your Python package.

### `docs/source/project.py`

Project name, author and `codename` (Python package name)

### LaTeX: `docs/source/project.sty`

Project name, institution, logo and aesthetics.

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

## edit below variables as per your requirements -
REPO_NAME = "ML Based Job Offers Recommender System"
AUTHOR_USER_NAME = "Khalil SAMBA"
SRC_REPO = "job_offers_recommender"
LIST_OF_REQUIREMENTS = [
    "streamlit",
    "numpy",
    "pandas",
    "scikit-learn",
    "notebook",
    "PyYAML"
]


setup(
    name=SRC_REPO,
    version="0.0.1",
    author=AUTHOR_USER_NAME,
    description="A small local packages for ML based job offers recommendations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ksambaa/ML-Based-JobOffers-Recommender-System",
    author_email="khaliilsambaa@gmail.com",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.7",
    install_requires=LIST_OF_REQUIREMENTS
)

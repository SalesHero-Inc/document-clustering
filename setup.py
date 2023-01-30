"""Document Clustering package setup"""

from setuptools import find_packages, setup

BASE_DEPENDENCIES = [
    "opencv-python-headless==4.5.5.64",
    "numpy==1.21.5",
    'typing-extensions; python_version == "4.1.*"',
]

TEST_DEPENDENCIES = ["pytest", "parameterized"]

CLUSTER_DETECTOR_DEPENDENCIES = [
    "enumerator==0.1.4",
    "pandas==1.3.5",
    "Pillow==9.2.0",
    "scikit-learn==1.0.2",
    "keybert==0.7.0",
    "sentence-transformers==2.2.2",
]

EXTRAS_DEPENDENCIES = {
    "test": TEST_DEPENDENCIES,
    "detector": CLUSTER_DETECTOR_DEPENDENCIES,
}
assert "all" not in EXTRAS_DEPENDENCIES, "'all' is a protected extras name."
EXTRAS_DEPENDENCIES["all"] = list(set(sum(EXTRAS_DEPENDENCIES.values(), [])))

setup(
    name="ah_document_cluster_extractor",
    version="0.0.3",
    author="Priyam Basu",
    author_email="priyam.basu@automationhero.ai",
    packages=find_packages(),
    python_requires=">=3.5, <3.10",
    install_requires=BASE_DEPENDENCIES,
    extras_require=EXTRAS_DEPENDENCIES,
)

from setuptools import setup, find_packages

setup(
    name="graphylovar",
    version="0.3.0",
    description=(
        "GraphyloVar: Predicting Functional Impact of Non-Coding Variants "
        "Using Multi-Species Evolutionary Graphs"
    ),
    author="Dongjoon Lim",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "tensorflow>=2.6",
        "spektral>=1.0",
        "networkx",
        "matplotlib",
        "tqdm",
        "pyyaml",
        "biopython",
    ],
    extras_require={
        "focal": ["focal-loss"],
        "liftover": ["pyliftover"],
        "all": ["focal-loss", "pyliftover"],
    },
    entry_points={
        "console_scripts": [
            "graphylovar-preprocess=scripts.preprocess:main",
            "graphylovar-train=scripts.train:main",
            "graphylovar-predict=scripts.predict:main",
            "graphylovar-predict-genome=scripts.predict_genome:main",
            "graphylovar-parse-maf=scripts.parse_maf:main",
        ],
    },
)

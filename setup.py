import setuptools
from pathlib import Path

setuptools.setup(
    name='gym_binpick',
    author="Seiok Kim",
    author_email="bboyseiok@deepest.ai",
    version='0.0.7',
    description="An OpenAI Gym Env for Partial Observability test",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(include="gym_binpick*"),
    include_package_data=True,
    install_requires=['gym', 'pybullet', 'numpy'],  # And any other dependencies foo needs
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
    python_requires='>=3.6'
)

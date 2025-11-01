from setuptools import setup, find_packages

# relative links to absolute
with open("./README.md", "r") as f:
    readme = f.read()


setup(
    name='image-to-image',
    version='0.3',
    packages=['image_to_image'],# find_packages(),
    install_requires=[
        # List any dependencies here, e.g. 'numpy', 'requests'
        "numpy", 
        "numba", 
        "matplotlib", 
        "opencv-python",
        "ipykernel", 
        "ipython", 
        "notebook", 
        "shapely", 
        "prime_printer", 
        "datasets==3.6.0",
        "scikit-image",
        "img-phy-sim",
        "pytorch-msssim",
        "kornia", 
        "mlflow", 
        "tensorboard",
        "tqdm",
        "torch",  # CPU fallback
        "torchvision"  # CPU fallback
    ],
    extras_require={  # pip install image-to-image[gpu]
        "gpu": [
            "torch==2.1.0+cu126",
            "torchvision==0.16.0+cu126",
            "torchaudio==2.1.0+cu126"
        ]
    },
    dependency_links=[
        "https://download.pytorch.org/whl/torch_stable.html"
    ],
    author="Tobia Ippolito",
    description = 'Image to Image translation with PyTorch Package.',
    long_description = readme,
    long_description_content_type="text/markdown",
    include_package_data=True,  # Ensures files from MANIFEST.in are included
    download_url = 'https://github.com/xXAI-botXx/Image-to-Image/archive/v_02.tar.gz',
    url="https://github.com/xXAI-botXx/Image-to-Image",
    project_urls={
        "Documentation": "https://xxai-botxx.github.io/Image-to-Image/image_to_image",
        "Source": "https://github.com/xXAI-botXx/Image-to-Image"
    },
    keywords = ['PyTorch', 'Computer-Vision', 'Physgen'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13'
    ],
)




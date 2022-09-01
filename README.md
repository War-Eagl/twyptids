# twyptids


## Installation

### Installing python packages 

Install the following packages 

```
pip install svgwrite
pip install svgpathtools
pip install cssutils
pip install numba
pip install torch-tools
pip install visdom
pip install ftfy
pip install git+https://github.com/openai/CLIP.git --no-deps
```

### Installing pydiffvg

- Clone this repository ```git clone https://github.com/War-Eagl/twyptids```
- ```cd twyptids```
- Clone the submodules ```git submodule update --init --recursive```
- ```cd diffvg```
- Run this python code inside diffvg folder:
 ```
 data = []
with open("CMakeLists.txt", "r+") as inFile:
    for line in inFile:
        if "find_package(TensorFlow)" in line:
            pass
        else:
            data.append(line)
    inFile.seek(0)
    for d in data:
        inFile.write(d)
    inFile.truncate()
  ```
- Run setup.py ```python setup.py install```

### Run

In the ```twyptids``` folder, run `modify_sketch.py`

```python modify_sketch.py```

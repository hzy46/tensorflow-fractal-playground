# tensorflow-fractal-playground
Use TensorFlow to generate beautiful fractals, including the Mandelbrot Set and the Julia set.

## Requirements:
- TensorFlow >= 1.0
- PIL

## Generate the Standard Mandelbrot Set:

Run:
```
python mandelbrot.py
```

and check "mandelbrot.png":

![](https://github.com/hzy46/tensorflow-fractal-playground/blob/master/img/m0.png?raw=true)

## Generate the Julia set:

Run:
```
python julia.py
```

and check "julia.png":

![](https://github.com/hzy46/tensorflow-fractal-playground/blob/master/img/j1.png?raw=true)

Modify the settings in julia.py to get a differnt Julia Set. The variable "c" is corresponding to the c of a Julia Set. The variable "bg_ratio" and "ratio" is used to set the colors.

Use the setting:
```
c = -0.8 * 1j
bg_ratio = (1, 3.5, 3.5)
ratio = (0.9, 0.9, 0.9)
```
Will get:

![](https://github.com/hzy46/tensorflow-fractal-playground/blob/master/img/j2.png?raw=true)


Use the setting:
```
c = 0.285 + 0.01 * 1j
bg_ratio = (4, 2.5, 1)
ratio = (2, 2, 0.1)
```
Will get:

![](https://github.com/hzy46/tensorflow-fractal-playground/blob/master/img/j3.png?raw=true)

Try more settings by yourself!

## Explore the Mandelbrot Set:

To get a local area image of Mandelbrot Set, run:

```
python mandelbrot_area.py
```

Check "mandelbrot.png":

![](https://github.com/hzy46/tensorflow-fractal-playground/blob/master/img/m1.png?raw=true)

Modify the varibales "start_x", "end_x", "start_y", "end_y" in mandelbrot_area.py to get a different area of the Mandelbrot Set. To get a different color, adjust the variables "ratio1, ratio2, ratio3".

For example, the following setting:

```
start_x = -0.090  # x range
end_x = -0.086
start_y = 0.654  # y range
end_y = 0.657
width = 1000
ratio1, ratio2, ratio3 = 0.2, 0.6, 0.6
```

will result in:

![](https://github.com/hzy46/tensorflow-fractal-playground/blob/master/img/m2.png?raw=true)

Setting:

```
start_x = -0.750  # x range
end_x = -0.747
start_y = 0.099  # y range
end_y = 0.102
width = 1000
ratio1, ratio2, ratio3 = 0.1, 0.1, 0.3
```

will generate:

![](https://github.com/hzy46/tensorflow-fractal-playground/blob/master/img/m3.png?raw=true)


## Image Size:

The variable "width" controls the width of generated image.

If memory is a issue, please decrease it.

## Jupyter Notebook:

You can use "Mandelbrot.ipynb" and "Julia.ipynb" for more convenient exploration. Enjoy it!


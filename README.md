
## Regenerate figures for IJCAS paper
This repository is used to generate the figures and data for the IJCAS submission titled
"Constrained Geometric Attitude Control on SO(3)" which is located at one of the following repos:

* [FDCL IJCAS Paper](https://github.com/fdcl-gwu/2016_IJCAS)
* [Shankar IJCAS Paper](https://github.com/skulumani/2016_IJCAS)

Both repos should be equivalent but at times I'll forget to push to one or the other.

## How to regenerate the data

The code is written in both Matlab and Python. 
Either will generate similar figures which are used in the manuscript. 

To regenerate the data all one needs to do is:

~~~
python generate_plots.py
~~~

Which will accomplish the following:

1. Instantiate a `spacecraft` object for each of the examples given in the paper
2. Simulate or read the experiemental data appropriately
3. Plot the data using `matplotlib`
4. Optionally save/write the figures to both PDF and PGF if desired by using the `-s` flag

## Modifying the figure size

All of the plotting commands in `generate_plots.py` accepts an input for the figure size.
This figure size is dependent on the size of the finish plot in a LaTeX document. 
You can modify this to increase/decrease the sizes of the figures, which is especially important when using PGF/Tikz.



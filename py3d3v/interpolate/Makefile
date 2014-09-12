
all : linear

linear : linear.pyx
	python setup.py build_ext --inplace

clean :
	rm -rf linear.c linear.so
	rm -rf *.pyc build

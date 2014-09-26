
all : interp.pyx tools.pyx
	python setup.py build_ext --inplace

clean :
	rm -rf interp.c interp.so
	rm -rf tools.c tools.so
	rm -rf *.pyc build

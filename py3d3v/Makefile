
all : interp.pyx _tools.pyx
	python setup.py build_ext --inplace

clean :
	rm -rf interp.c interp.so
	rm -rf _tools.c _tools.so
	rm -rf *.pyc build

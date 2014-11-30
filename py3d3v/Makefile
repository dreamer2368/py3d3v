
all : interp.pyx tools.pyx
	python setup.py build_ext --inplace

clean :
	rm -rf interp.cpp interp.so
	rm -rf tools.cpp tools.so
	rm -rf *.pyc build

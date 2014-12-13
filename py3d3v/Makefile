
all : interp.pyx tools.pyx ewald.pyx
	python setup.py build_ext --inplace

clean :
	rm -rf interp.cpp interp.so
	rm -rf tools.cpp tools.so
	rm -rf ewald.cpp ewald.so
	rm -rf *.pyc build

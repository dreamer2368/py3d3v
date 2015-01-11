
all : interp.pyx core.pyx ewald.pyx
	python setup.py build_ext --inplace

clean :
	rm -rf interp.cpp interp.so
	rm -rf core.cpp core.so
	rm -rf ewald.cpp ewald.so
	rm -rf *.pyc build

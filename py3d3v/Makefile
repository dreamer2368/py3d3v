
all : interp.pyx core.pyx ewald.pyx solvers.pyx
	python setup.py build_ext --inplace

clean :
	rm -f interp.cpp interp.so
	rm -f core.cpp core.so
	rm -f ewald.cpp ewald.so
	rm -f solvers.cpp solvers.so
	rm -rf *.pyc build
	rm -f *.pdf

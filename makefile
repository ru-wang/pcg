all: curve powells

curve: linear_solver.h conjugate_gradient.h curve.cc
	g++ curve.cc -o curve -std=c++17 -isystem /usr/include/eigen3 \
		-Ofast -mavx -mfma -DNDEBUG_LM -DNDEBUG_CG

powells: linear_solver.h conjugate_gradient.h powells.cc
	g++ powells.cc -o powells -std=c++17 -isystem /usr/include/eigen3 \
		-Ofast -mavx -mfma -DNDEBUG_LM -DNDEBUG_CG

clean:
	rm -f curve powells

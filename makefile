all: linear_solver.h conjugate_gradient.h powells.cc
	g++ powells.cc -o powells -std=c++17 -isystem /usr/include/eigen3 \
		-Ofast -mavx -mfma -DNDEBUG_LM -DNDEBUG_CG

clean:
	rm powells

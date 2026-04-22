ENV_NAME=extreme-rainfall-py311
BUILD_DIR=build
FORTRAN_DIR=fortran

.PHONY: help env-check fortran-smoke fortran-demo run-fortran-demo clean

help:
	@echo "Targets:"
	@echo "  env-check         Check core Python imports"
	@echo "  fortran-smoke     Compile and run a trivial Fortran program"
	@echo "  fortran-demo      Compile rolling_accumulation.f90"
	@echo "  run-fortran-demo  Run rolling_accumulation on demo input"
	@echo "  clean             Remove build artifacts"

env-check:
	python -c "import numpy, pandas, xarray, scipy, matplotlib, cartopy, pyarrow, requests, yaml, netCDF4, cfgrib; from netCDF4 import Dataset; print('python env ok')"

fortran-smoke:
	mkdir -p $(BUILD_DIR)
	printf '%s\n' \
	'program hello_fortran' \
	'  implicit none' \
	'  print *, "Fortran toolchain OK"' \
	'end program hello_fortran' > $(BUILD_DIR)/hello_fortran.f90
	gfortran -O2 -o $(BUILD_DIR)/hello_fortran $(BUILD_DIR)/hello_fortran.f90
	./$(BUILD_DIR)/hello_fortran

fortran-demo:
	mkdir -p $(BUILD_DIR)
	gfortran -O2 -o $(BUILD_DIR)/rolling_accumulation $(FORTRAN_DIR)/rolling_accum.f90

run-fortran-demo: fortran-demo
	./$(BUILD_DIR)/rolling_accumulation $(FORTRAN_DIR)/demo_input.txt $(FORTRAN_DIR)/demo_output.txt
	cat $(FORTRAN_DIR)/demo_output.txt

clean:
	rm -rf $(BUILD_DIR)
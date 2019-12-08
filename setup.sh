# install z3 with bindings
# there is no official z3 PyPi
cd ~
git clone https://github.com/Z3Prover/z3

cd z3/

python scripts/mk_make.py --python
cd build
make -j4
make install

python -c 'import z3; print(z3.get_version_string())'

import os
import sys

# Compile the TensorFlow ops.
compile_command = ("g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 ./mcpm/util/tf_ops/vec_to_tri.cc "
                   "./mcpm/util/tf_ops/tri_to_vec.cc -o ./mcpm/util/tf_ops/matpackops.so "
                   "-fPIC -I $(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')")

if sys.platform == "darwin":
    compile_command += " -undefined dynamic_lookup"

os.system(compile_command)

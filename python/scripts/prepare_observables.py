import time

from sym_contractions import build_observables

###################################################################
# Parameters
###################################################################
labels = ["XXXX_p2341"]  # interaction terms
Lambdas = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # cutoff
mass = 0.5
###################################################################

# Load or build and save observables, then build Hamiltonian for each Lambda
for Lambda in Lambdas:
    print(f"Processing Lambda={Lambda}...")
    start_time = time.time()
    observables = build_observables(
        labels,
        Lambda=Lambda,
        mass=mass,
        clean=False,
        verbose=True,
        save=True,
        storage="sparse",
        parallel=True,
    )
    elapsed_time = time.time() - start_time
    print(f"Done in {elapsed_time:.2f} seconds.")
